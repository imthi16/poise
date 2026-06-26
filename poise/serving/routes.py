"""FastAPI routes (CLAUDE.md §7). Base prefix ``/v1``; JSON in/out."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    mode: Optional[str] = None              # static | pid | ppo (default: env mode)
    static_depth: Optional[int] = None      # only when mode=static
    return_trace: bool = True


class ModeRequest(BaseModel):
    mode: str


class RagRequest(BaseModel):
    query: str
    k: int = Field(default=4, ge=1, le=50)


def _service(request: Request):
    return request.app.state.service


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/generate")
    def generate(req: GenerateRequest, request: Request):
        svc = _service(request)
        mode = req.mode or svc.mode
        if mode not in ("static", "pid", "ppo"):
            raise HTTPException(status_code=400, detail=f"unknown mode {mode!r}")
        if mode == "ppo" and not svc.policy_loaded:
            # allowed: allocator falls back to PID, but be explicit about it
            pass
        return svc.generate(
            req.prompt, req.max_new_tokens, mode=mode,
            static_depth=req.static_depth, return_trace=req.return_trace,
        )

    @router.get("/v1/telemetry")
    def telemetry(request: Request):
        return _service(request).telemetry()

    @router.get("/v1/state")
    def state(request: Request):
        return _service(request).state()

    @router.post("/v1/config/mode")
    def set_mode(req: ModeRequest, request: Request):
        ok, msg = _service(request).set_mode(req.mode)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return {"mode": req.mode, "ok": True}

    @router.get("/v1/runs/{run_id}")
    def get_run(run_id: str, request: Request):
        run = _service(request).get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @router.post("/v1/rag/query")
    def rag_query(req: RagRequest, request: Request):
        return _service(request).rag_query(req.query, k=req.k)

    @router.get("/metrics")
    def metrics(request: Request):
        svc = _service(request)
        # refresh telemetry-derived gauges before scraping
        try:
            svc.telemetry()
        except Exception:
            pass
        body, content_type = svc.metrics.render()
        return Response(content=body, media_type=content_type)

    @router.get("/health")
    def health(request: Request):
        svc = _service(request)
        return {"status": "ok", "device": svc.cfg.model.device, "mode": svc.mode}

    return router


__all__ = ["build_router", "GenerateRequest", "ModeRequest", "RagRequest"]
