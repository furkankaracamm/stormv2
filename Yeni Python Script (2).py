import os
import sys
import time
import json
import logging
import asyncio
import aiohttp
import heapq
import random
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Literal, Union, Any
from enum import Enum
import nest_asyncio

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    print("CRITICAL: `pip install pydantic` gerekli.")
    sys.exit(1)

# --- CONFIGURATION ---
BASE_DIR = Path.home() / ".gemini" / "antigravity" / "scratch"
WORK_DIR = BASE_DIR
LOG_DIR = WORK_DIR / "logs_supervisor"
THESIS_DIR = WORK_DIR / "thesis_output"
MEMORY_PATH = WORK_DIR / "unified_memory.json"
SYSTEM_MODE = "RIGOROUS"
DRIFT_THRESHOLD = 75
MAX_RECURSION = 3
LLM_TIMEOUT = 120

for d in [LOG_DIR, THESIS_DIR]: d.mkdir(parents=True, exist_ok=True)
nest_asyncio.apply()

# --- ONTOLOGY ---
class TheoreticalDeadlock(BaseModel):
    description: str
    core_axiom: str
    deadlock_reason: str
    status: str = "ACTIVE"
    integrity_score: int = 100

class UndecidabilityVerdict(BaseModel):
    is_solvable: bool
    reason: str
    category: Literal["EMPIRICAL", "METAPHYSICAL", "DATA_VOID", "SOLVABLE"]

class DriftReport(BaseModel):
    original_core: str
    current_state: str
    semantic_distance: str
    integrity_retained: int
    verdict: Literal["STABLE", "DRIFT", "CAPITULATION"]

class DecisionFork(BaseModel):
    topic: str
    path_A: Dict[str, str]
    path_B: Dict[str, str]
    epistemic_tradeoff: str
    recommendation: Literal["PATH_A", "PATH_B", "AGNOSTIC"]

# --- INFRASTRUCTURE ---
class UnifiedMemory:
    def __init__(self):
        self.data = {"constraints": [], "undecidables": []}
        if MEMORY_PATH.exists():
            try:
                with open(MEMORY_PATH, "r", encoding="utf-8") as f: self.data = json.load(f)
            except: pass

    def get_context(self, topic: str) -> str:
        rels = [c for c in self.data["constraints"] if c["topic"] in topic]
        unds = [u for u in self.data["undecidables"] if u["topic"] in topic]
        return f"CONSTRAINTS: {json.dumps(rels)}\nUNDECIDABLES: {json.dumps(unds)}"

    def learn_failure(self, topic: str, reason: str, category: str):
        entry = {"topic": topic, "reason": reason, "category": category, "timestamp": str(datetime.now())}
        target = self.data["undecidables"] if category == "UNDECIDABLE" else self.data["constraints"]
        target.append(entry)
        with open(MEMORY_PATH, "w", encoding="utf-8") as f: json.dump(self.data, f, indent=2)

MEMORY = UnifiedMemory()

class ServiceGateway:
    def __init__(self):
        self._session = None
        self._semaphore = asyncio.Semaphore(5)

    async def get_session(self):
        if not self._session or self._session.closed: self._session = aiohttp.ClientSession()
        return self._session

    async def call_llm(self, prompt: str, system_persona: str, schema: Optional[BaseModel] = None) -> Any:
        async with self._semaphore:
            session = await self.get_session()
            payload = {
                "model": "llama3.3", "prompt": f"SYSTEM: {system_persona}\nUSER: {prompt}",
                "stream": False, "format": "json" if schema else None,
                "options": {"temperature": 0.7 if SYSTEM_MODE == "EXPLORATORY" else 0.4}
            }
            try:
                async with session.post("http://localhost:11434/api/generate", json=payload, timeout=LLM_TIMEOUT) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get("response", "")
                        if schema: 
                            try: return json.loads(content.replace("```json", "").replace("```", "").strip())
                            except: return None
                        return content
            except Exception as e: print(f"LLM Error: {e}")
            return None
    
    async def close(self):
        if self._session: await self._session.close()

GATEWAY = ServiceGateway()

# --- SUPERVISOR MODULES ---
class UndecidabilityOracle:
    async def check_solvability(self, deadlock: TheoreticalDeadlock) -> UndecidabilityVerdict:
        prompt = f"DEADLOCK: {deadlock.description}\nIs this empirically solvable or metaphysical?"
        res = await GATEWAY.call_llm(prompt, "You are a Philosopher of Science.", schema=True)
        if res: return UndecidabilityVerdict(**res)
        return UndecidabilityVerdict(is_solvable=True, reason="Default", category="SOLVABLE")

class DriftMonitor:
    async def check_drift(self, original: TheoreticalDeadlock, current: TheoreticalDeadlock) -> DriftReport:
        prompt = f"ORIGINAL: {original.core_axiom}\nCURRENT: {current.description}\nDid we capitulate?"
        res = await GATEWAY.call_llm(prompt, "You are a Theoretical Purist.", schema=True)
        if res: return DriftReport(**res)
        return DriftReport(original_core="", current_state="", semantic_distance="Error", integrity_retained=0, verdict="CAPITULATION")

class AgnosticForkGenerator:
    async def generate(self, deadlock: TheoreticalDeadlock) -> DecisionFork:
        prompt = f"DEADLOCK: {deadlock.description}\nCreate a Decision Fork (Path A vs B). Return AGNOSTIC if value-based."
        res = await GATEWAY.call_llm(prompt, "You are a Neutral Arbiter.", schema=True)
        if res: return DecisionFork(**res)
        return None

# --- MAIN LOOP ---
class ResearchSupervisor:
    async def conduct_research(self, topic: str):
        print(f"\n👁️ [SUPERVISOR] Watching: {topic}")
        context = MEMORY.get_context(topic)
        
        prompt = f"TOPIC: {topic}\nCONTEXT: {context}\nIdentify a Theoretical Deadlock."
        res = await GATEWAY.call_llm(prompt, "You are a Senior Theorist.", schema=True)
        if not res: return
        
        try: deadlock = TheoreticalDeadlock(description=res.get("description"), core_axiom=res.get("core_axiom"), deadlock_reason=res.get("deadlock_reason"))
        except: return

        print(f"   🔒 Deadlock: {deadlock.description[:60]}...")
        
        oracle = UndecidabilityOracle()
        verdict = await oracle.check_solvability(deadlock)
        if not verdict.is_solvable:
            print(f"   🛑 HALT: {verdict.category} - {verdict.reason}")
            MEMORY.learn_failure(topic, verdict.reason, "UNDECIDABLE")
            return

        # Recursive Loop Simulation
        monitor = DriftMonitor()
        for i in range(MAX_RECURSION):
            # Simulation: Check drift (In reality, methods reshape theory here)
            report = await monitor.check_drift(deadlock, deadlock)
            if report.integrity_retained < DRIFT_THRESHOLD:
                print(f"   ⚠️ DRIFT: {report.verdict}"); return

        gen = AgnosticForkGenerator()
        fork = await gen.generate(deadlock)
        if fork:
            print(f"   ⚖️ Recommendation: {fork.recommendation}")
            with open(THESIS_DIR / f"FORK_{int(time.time())}.json", "w") as f: json.dump(fork.model_dump(), f, indent=2)

async def main():
    print(">>> STORM v24: THE EPISTEMIC SUPERVISOR <<<")
    supervisor = ResearchSupervisor()
    topics = ["The ontology of deleted bots", "Algorithmic polarization"]
    tasks = [supervisor.conduct_research(t) for t in topics]
    await asyncio.gather(*tasks)
    await GATEWAY.close()

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
