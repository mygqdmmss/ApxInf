use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use crate::server::service::{HttpResponse, ProtocolRuntime, ProtocolService};

pub const GENERATE_PATH: &str = "/v1/evaluations/generate";
const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 1024 * 1024;

pub fn serve<S: ProtocolRuntime + 'static>(
    listener: TcpListener,
    service: Arc<ProtocolService<S>>,
) -> std::io::Result<()> {
    for connection in listener.incoming() {
        match connection {
            Ok(stream) => {
                let service = Arc::clone(&service);
                std::thread::spawn(move || {
                    let _ = handle_connection(stream, &service);
                });
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

pub fn handle_connection<S: ProtocolRuntime>(
    mut stream: TcpStream,
    service: &ProtocolService<S>,
) -> std::io::Result<()> {
    let request = read_request(&mut stream)?;
    if request.method == "GET" && request.path == "/health" {
        return write_response(&mut stream, service.health_response());
    }
    if request.method != "POST" || request.path != GENERATE_PATH {
        return write_response(
            &mut stream,
            HttpResponse {
                status: 404,
                content_type: "application/json",
                body: br#"{"error":{"type":"not_found","message":"route not found"}}"#.to_vec(),
            },
        );
    }

    let wants_stream = serde_json::from_slice::<serde_json::Value>(&request.body)
        .ok()
        .and_then(|value| value.get("stream").and_then(serde_json::Value::as_bool))
        .unwrap_or(false);
    if !wants_stream {
        return write_response(&mut stream, service.handle_non_stream(&request.body));
    }

    let mut generation = match service.start_stream(&request.body) {
        Ok(generation) => generation,
        Err(response) => return write_response(&mut stream, response),
    };
    let first_frame = match generation.next_frame() {
        Ok(Some(frame)) => frame,
        Ok(None) => return write_response(&mut stream, empty_response()),
        Err(error) => {
            return write_response(&mut stream, crate::server::service::runtime_response(error))
        }
    };
    let headers = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if stream
        .write_all(headers)
        .and_then(|_| stream.flush())
        .is_err()
        || stream
            .write_all(&first_frame)
            .and_then(|_| stream.flush())
            .is_err()
    {
        generation.cancel();
        return Ok(());
    }
    loop {
        match generation.next_frame() {
            Ok(Some(frame)) => {
                if stream
                    .write_all(&frame)
                    .and_then(|_| stream.flush())
                    .is_err()
                {
                    generation.cancel();
                    return Ok(());
                }
            }
            Ok(None) => return Ok(()),
            Err(error) => {
                let frame =
                    crate::server::response::sse_error_frame(generation.request_id(), &error);
                let _ = stream.write_all(&frame).and_then(|_| stream.flush());
                generation.cancel();
                return Ok(());
            }
        }
    }
}

fn empty_response() -> HttpResponse {
    HttpResponse {
        status: 500,
        content_type: "application/json",
        body: br#"{"error":{"type":"runtime_error","message":"empty stream"}}"#.to_vec(),
    }
}

fn write_response(stream: &mut TcpStream, response: HttpResponse) -> std::io::Result<()> {
    let status = match response.status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        499 => "Client Closed Request",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        status,
        response.content_type,
        response.body.len()
    )?;
    stream.write_all(&response.body)?;
    stream.flush()
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<Request> {
    let mut buffer = Vec::with_capacity(4096);
    let header_end = loop {
        let mut chunk = [0u8; 1024];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "request ended",
            ));
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.len() > MAX_HEADER_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "headers too large",
            ));
        }
        if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break index;
        }
    };
    let header_bytes = &buffer[..header_end];
    let header_text = std::str::from_utf8(header_bytes).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "headers are not UTF-8")
    })?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing request line")
    })?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or_default().to_owned();
    let content_length = lines
        .filter_map(|line| line.split_once(':'))
        .find_map(|(name, value)| {
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim())
        })
        .unwrap_or("0")
        .parse::<usize>()
        .map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid content length")
        })?;
    if content_length > MAX_BODY_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "body too large",
        ));
    }
    let body_start = header_end + 4;
    let mut body = buffer.get(body_start..).unwrap_or_default().to_vec();
    while body.len() < content_length {
        let remaining = content_length - body.len();
        let mut chunk = vec![0u8; remaining.min(8192)];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "body ended",
            ));
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(content_length);
    Ok(Request { method, path, body })
}

#[cfg(test)]
mod tests {
    use super::{handle_connection, read_request};
    use crate::server::service::{ProtocolRuntime, TokenStream};
    use apxinf_model::{RuntimeCapabilities, RuntimeError, RuntimeRequest};
    use std::io::{Read, Write};
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    struct FakeRuntime {
        cancelled: Arc<Mutex<bool>>,
        token_count: usize,
        max_model_len: usize,
        yielded: Arc<AtomicUsize>,
        warmup_fails: bool,
    }

    struct FakeStream {
        remaining: usize,
        cancelled: Arc<Mutex<bool>>,
        yielded: Arc<AtomicUsize>,
    }

    impl TokenStream for FakeStream {
        fn next_token(&mut self) -> Result<Option<u32>, RuntimeError> {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;
            self.yielded.fetch_add(1, Ordering::Relaxed);
            Ok(Some(7))
        }

        fn cancel(&self) {
            *self.cancelled.lock().unwrap() = true;
        }
    }

    impl ProtocolRuntime for FakeRuntime {
        fn capabilities(&self) -> RuntimeCapabilities {
            RuntimeCapabilities::frozen_qwen35(self.max_model_len, 0)
        }

        fn start(&self, _request: RuntimeRequest) -> Result<Box<dyn TokenStream>, RuntimeError> {
            Ok(Box::new(FakeStream {
                remaining: self.token_count,
                cancelled: Arc::clone(&self.cancelled),
                yielded: Arc::clone(&self.yielded),
            }))
        }

        fn warmup(&self) -> Result<(), RuntimeError> {
            if self.warmup_fails {
                Err(RuntimeError::Execution("warmup failed".into()))
            } else {
                Ok(())
            }
        }
    }

    fn request(method: &str, path: &str, body: &[u8]) -> Vec<u8> {
        format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        )
        .into_bytes()
    }

    fn exchange(raw: Vec<u8>) -> Vec<u8> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let service = Arc::new(crate::server::service::ProtocolService::new(
            FakeRuntime {
                cancelled: Arc::new(Mutex::new(false)),
                token_count: 2,
                max_model_len: 32,
                yielded: Arc::new(AtomicUsize::new(0)),
                warmup_fails: false,
            },
            true,
        ));
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, &service).unwrap();
        });
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(&raw).unwrap();
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        server.join().unwrap();
        response
    }

    #[test]
    fn health_and_not_found_have_http_statuses() {
        let health = exchange(request("GET", "/health", b""));
        assert!(health.starts_with(b"HTTP/1.1 200 OK\r\n"));
        assert!(String::from_utf8_lossy(&health).contains("\"status\":\"ok\""));
        let missing = exchange(request("GET", "/missing", b""));
        assert!(missing.starts_with(b"HTTP/1.1 404 Not Found\r\n"));
    }

    #[test]
    fn unhealthy_service_returns_http_503_from_health() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let service = Arc::new(crate::server::service::ProtocolService::new(
            FakeRuntime {
                cancelled: Arc::new(Mutex::new(false)),
                token_count: 0,
                max_model_len: 32,
                yielded: Arc::new(AtomicUsize::new(0)),
                warmup_fails: true,
            },
            false,
        ));
        service.mark_unhealthy();
        let server_service = Arc::clone(&service);
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, &server_service).unwrap();
        });
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(&request("GET", "/health", b"")).unwrap();
        client.shutdown(Shutdown::Write).unwrap();
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        server.join().unwrap();
        assert!(response.starts_with(b"HTTP/1.1 503 Service Unavailable\r\n"));
        assert!(String::from_utf8_lossy(&response).contains("\"status\":\"unhealthy\""));
    }

    #[test]
    fn malformed_and_structured_requests_are_json_400() {
        let malformed = exchange(request("POST", "/v1/evaluations/generate", b"{not-json"));
        assert!(malformed.starts_with(b"HTTP/1.1 400 Bad Request\r\n"));
        assert!(String::from_utf8_lossy(&malformed).contains("\"error\""));
        let invalid = br#"{"input_ids":[],"max_new_tokens":1,"temperature":0,"ignore_eos":true,"stream":false}"#;
        let invalid = exchange(request("POST", "/v1/evaluations/generate", invalid));
        assert!(invalid.starts_with(b"HTTP/1.1 400 Bad Request\r\n"));
        assert!(String::from_utf8_lossy(&invalid).contains("\"error\""));
    }

    #[test]
    fn json_and_sse_success_are_incremental_protocols() {
        let body = br#"{"input_ids":[1,2,3,4,5,6,7,8],"max_new_tokens":1,"temperature":0,"ignore_eos":true,"stream":false}"#;
        let json = exchange(request("POST", "/v1/evaluations/generate", body));
        assert!(json.starts_with(b"HTTP/1.1 200 OK\r\n"));
        assert!(String::from_utf8_lossy(&json).contains("\"completion_tokens\":1"));
        let stream_body = br#"{"input_ids":[1,2,3,4,5,6,7,8],"max_new_tokens":2,"temperature":0,"ignore_eos":true,"stream":true}"#;
        let sse = exchange(request("POST", "/v1/evaluations/generate", stream_body));
        let sse_text = String::from_utf8_lossy(&sse);
        assert!(sse.starts_with(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"));
        assert!(sse_text.contains("\"index\":0"));
        assert!(sse_text.contains("\"index\":1"));
        assert!(sse_text.contains("data: [DONE]\n\n"));
    }

    #[test]
    fn disconnect_after_first_sse_frame_cancels_real_stream() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let cancelled = Arc::new(Mutex::new(false));
        let yielded = Arc::new(AtomicUsize::new(0));
        let service = Arc::new(crate::server::service::ProtocolService::new(
            FakeRuntime {
                cancelled: Arc::clone(&cancelled),
                token_count: 20_000,
                max_model_len: 20_008,
                yielded: Arc::clone(&yielded),
                warmup_fails: false,
            },
            true,
        ));
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, &service).unwrap();
        });
        let mut client = TcpStream::connect(address).unwrap();
        let body = br#"{"input_ids":[1,2,3,4,5,6,7,8],"max_new_tokens":20000,"temperature":0,"ignore_eos":true,"stream":true}"#;
        client
            .write_all(&request("POST", "/v1/evaluations/generate", body))
            .unwrap();
        let mut first = [0u8; 256];
        let _ = client.read(&mut first).unwrap();
        client.shutdown(Shutdown::Both).unwrap();
        drop(client);
        server.join().unwrap();
        assert!(*cancelled.lock().unwrap());
        assert!(yielded.load(Ordering::Relaxed) < 20_000);
    }

    #[test]
    fn request_parser_reads_content_length_without_overread() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_request(&mut stream).unwrap();
            assert_eq!(request.body, b"abc");
        });
        let mut client = TcpStream::connect(address).unwrap();
        client
            .write_all(b"POST /x HTTP/1.1\r\nContent-Length: 3\r\n\r\nabcdef")
            .unwrap();
        drop(client);
        server.join().unwrap();
    }
}
