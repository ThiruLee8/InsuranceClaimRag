.PHONY: up down down-v logs backend-shell migrate test-backend seed-env

up:
	docker compose up --build

down:
	docker compose down

down-v:
	@echo "WARNING: This removes all persistent volumes (SQL, Azurite, Chroma, Ollama)."
	docker compose down -v

logs:
	docker compose logs -f

backend-shell:
	docker compose exec backend bash

migrate:
	docker compose exec backend alembic upgrade head

test-backend:
	docker compose exec backend pytest -q

seed-env:
	@if not exist .env copy .env.example .env
	@echo ".env ready (Windows). Edit SQL_SERVER_PASSWORD if needed."
