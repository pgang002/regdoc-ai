from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/day10_final/deployment_validation.json"


def main() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = compose.get("services", {})
    required = {"postgres", "redis", "api", "worker", "ui", "prometheus"}
    missing = sorted(required - set(services))
    api = services.get("api", {})
    checks = {
        "compose_yaml_valid": isinstance(compose, dict),
        "required_services_present": not missing,
        "missing_services": missing,
        "api_healthcheck_present": bool(api.get("healthcheck")),
        "prometheus_config_present": (ROOT / "configs/prometheus.yml").exists(),
        "ci_workflow_present": (ROOT / ".github/workflows/ci.yml").exists(),
        "dockerfile_present": (ROOT / "Dockerfile").exists(),
        "docker_cli_available": shutil.which("docker") is not None,
    }
    checks["static_configuration_passed"] = all(
        bool(checks[key])
        for key in (
            "compose_yaml_valid",
            "required_services_present",
            "api_healthcheck_present",
            "prometheus_config_present",
            "ci_workflow_present",
            "dockerfile_present",
        )
    )
    checks["execution_note"] = (
        "Docker CLI unavailable in this runtime; configuration was parsed and statically validated, "
        "but containers were not launched."
        if not checks["docker_cli_available"]
        else "Docker CLI is available; run docker compose up --build for service validation."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
