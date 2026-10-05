# grid2op-harness — dev/ops targets
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VENV := $(ROOT)/.venv/bin
PY   := $(VENV)/python
PORT := 8731

.PHONY: help backend web mock serve-all test test-a test-b test-c test-d test-e eval fixtures clean

help:
	@echo "  make backend   FastAPI app (sim + opencode driver) on :$(PORT)"
	@echo "  make web       Vite dev server on :5173 (proxies to :$(PORT))"
	@echo "  make mock      mock SIM-API + recorded SSE on :$(PORT) (offline)"
	@echo "  make serve-all backend + web together"
	@echo "  make test      all workstream suites"
	@echo "  make eval      run ONE real episode (v0 gate)"
	@echo "  make fixtures  regenerate contracts/examples from live grid2op"

backend:
	cd $(ROOT) && $(VENV)/uvicorn backend.app.main:app --host 127.0.0.1 --port $(PORT) --reload

web:
	cd $(ROOT)/web && npm run dev

mock:
	cd $(ROOT) && python3 contracts/mock/mock_sim_server.py --port $(PORT)

serve-all:
	$(ROOT)/scripts/serve_all.sh

test: test-a test-b test-c test-d test-e

test-a:
	cd $(ROOT) && $(PY) -m pytest backend/tests -q
test-b:
	cd $(ROOT) && $(PY) -m pytest cli/tests -q
test-c:
	@echo "(WS C is docs; run its self-checks)"; cd $(ROOT) && grep -rn "env.grid" docs AGENTS.md recipes || echo "  OK: no env.grid references"
test-d:
	cd $(ROOT)/web && npm test -- --run
test-e:
	cd $(ROOT) && $(PY) -m pytest tests/integration -q

eval:
	cd $(ROOT) && $(PY) tests/integration/run_episode.py

fixtures:
	cd $(ROOT) && $(PY) scripts/gen_fixtures.py

# ---- benchmark ----
.PHONY: bench-validate bench-baselines bench-pilot bench-report
bench-validate:
	cd $(ROOT) && $(PY) bench/validate_metric.py
bench-baselines:
	cd $(ROOT) && $(PY) bench/run_baselines.py --standard
bench-pilot:
	cd $(ROOT) && $(PY) bench/run_pilot.py --conc 2
bench-pilot-1:
	cd $(ROOT) && $(PY) bench/run_llm.py --chronic 0 --horizon 96 --port 8820
bench-report:
	cd $(ROOT) && $(PY) bench/report.py --baselines $(wildcard runs/bench-*/baselines.jsonl) --llm $(wildcard runs/llm-*/results.json) --out runs/leaderboard

clean:
	rm -rf runs/eval-* runs/llm-* runs/bench-* runs/backend-*.log web/dist web/node_modules/.vite
