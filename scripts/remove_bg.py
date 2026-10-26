#!/usr/bin/env python3
"""AI 抠图脚本：从 stdin 读入图片，输出透明背景 PNG 到 stdout"""
import sys
from rembg import remove

def main():
    input_data = sys.stdin.buffer.read()
    output_data = remove(input_data)
    sys.stdout.buffer.write(output_data)

if __name__ == "__main__":
    main()
