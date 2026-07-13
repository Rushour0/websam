.PHONY: help install dev studio demo build test typecheck clean

SHELL := /bin/bash
# Load nvm and switch to the .nvmrc-pinned Node version before every recipe,
# since pnpm hard-fails its engine check against whatever Node happens to be active.
NVM_USE := export NVM_DIR="$$HOME/.nvm"; [ -s "$$NVM_DIR/nvm.sh" ] && source "$$NVM_DIR/nvm.sh" && nvm use >/dev/null;

help:
	@echo "Targets:"
	@echo "  make install   - install dependencies (pnpm)"
	@echo "  make studio    - run the studio app dev server"
	@echo "  make demo      - run the demo app dev server"
	@echo "  make dev       - alias for 'make studio'"
	@echo "  make build     - build all packages"
	@echo "  make test      - run package tests"
	@echo "  make typecheck - typecheck the workspace"
	@echo "  make clean     - remove node_modules and build output"

install:
	$(NVM_USE) pnpm install

dev: studio

studio: install
	$(NVM_USE) pnpm --filter websam-studio dev

demo: install
	$(NVM_USE) pnpm --filter websam-demo dev

build:
	$(NVM_USE) pnpm build

test:
	$(NVM_USE) pnpm test

typecheck:
	$(NVM_USE) pnpm typecheck

clean:
	pnpm -r exec rm -rf node_modules dist
	rm -rf node_modules
