#!/usr/bin/env python3
"""Build the defense deck for 项目6 - Agent for Kernel.

每一页都是一张完整设计的手绘信息图（AI 生成，逐页人工校对文字与数字），
风格对齐 Agent4System-0820.pdf：米色纸底、墨线手绘、橙/青强调色。
数据口径：REPORT.md（commit 06993a2 及其上的报告提交）。

页面清单（pdf/assets/s*.png，16:9 全幅）：
  s01_cover      封面：4090 能飞 · 飞行显卡主视觉
  s02_scope      从零开始：64 层混合架构 + W4A16 + 合同约束
  s03_results    成果总览：47× / 2.3× / 100% / 23.5 tok/s + 门禁徽章
  s04_layer1     第一层：六道门禁 + 断连容量泄漏的发现与修复
  s05_waterfall  第二层：16K 首 token 瀑布图（1635 → 34.6 s）
  s06_softmax    深潜 1：softmax ×512 冗余扫描（修复前后对比图）
  s07_bottleneck 深潜 2：GEMV 退化 + cudaMalloc 占 80%（甜甜圈图）
  s08_method     实验方法：四卡并行 / 三变体归因 / 位相等 / 双副本复测
  s09_mm         bonus 1 多模态：六站流水线，每站有判据
  s10_c4         bonus 2 C4：调度流图 + 官方校准记分卡
  s11_limits     取舍与边界：四组取舍 + 不作声称
  s12_repro      可复现：终端卡 + 验收 SHA + 回滚开关
  s13_team       分工说明（依据 git 提交记录）
  s14_closing    收尾：为什么叫「4090 能飞」
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

import os

AUTHOR = os.environ.get("PPT_AUTHOR", "程仁龙")
WORK = "4090能飞"
OUT = Path(f"/mnt/chuangxin/team2/ApxInf/pdf/项目6_{AUTHOR}_{WORK}_v1.pptx")
ASSETS = Path("/mnt/chuangxin/team2/ApxInf/pdf/assets")

W, H = Inches(13.333), Inches(7.5)

PAGES = [
    "s01_cover.png",
    "s02_scope.png",
    "s03_results.png",
    "s04_layer1.png",
    "s05_waterfall.png",
    "s06_softmax.png",
    "s07_bottleneck.png",
    "s08_method.png",
    "s09_mm.png",
    "s10_c4.png",
    "s11_limits.png",
    "s12_repro.png",
    "s13_team.png",
    "s14_closing.png",
]


def build() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    for name in PAGES:
        path = ASSETS / name
        assert path.exists(), f"missing page art: {path}"
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.shapes.add_picture(str(path), 0, 0, W, H)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"wrote {OUT} ({len(PAGES)} slides)")


if __name__ == "__main__":
    build()
