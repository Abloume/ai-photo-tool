# ai-photo-tool 开发/生产入口
# 用法:
#   make dev     开发模式：air 热重载，改 .go 自动编译重启
#   make build   编译生产二进制到 bin/
#   make run     编译并直接运行
#   make test    跑测试
#   make clean   清理 tmp/ 和 bin/

# air 装在 GOPATH/bin，直接用绝对路径，不依赖用户 PATH 配置
GOBIN ?= $(shell go env GOPATH)/bin

.PHONY: dev build run test clean

dev:
	$(GOBIN)/air

build:
	@mkdir -p bin
	go build -o bin/ai-photo-tool .

run: build
	./bin/ai-photo-tool

test:
	go test ./...

clean:
	rm -rf tmp bin
