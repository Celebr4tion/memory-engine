"""
Health check endpoints for web applications and APIs.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)


class HealthEndpoints:
    """Provides health check endpoints for web frameworks."""

    def __init__(self, health_checker, service_monitor, metrics_collector=None):
        self.health_checker = health_checker
        self.service_monitor = service_monitor
        self.metrics_collector = metrics_collector
        self._startup_time = time.time()

    async def health_check(self, include_details: bool = False) -> Dict[str, Any]:
        """Basic health check endpoint."""
        try:
            results = await self.health_checker.run_all_checks()
            overall_status = self.health_checker.get_overall_status()

            response = {
                "status": overall_status.value,
                "timestamp": time.time(),
                "uptime": time.time() - self._startup_time,
                "checks": len(results),
            }

            if include_details:
                response["details"] = {
                    name: {
                        "status": result.status.value,
                        "message": result.message,
                        "response_time": result.response_time,
                    }
                    for name, result in results.items()
                }

            return response

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "timestamp": time.time(), "error": str(e)}

    async def readiness_check(self) -> Dict[str, Any]:
        """Readiness check - determines if service is ready to accept traffic."""
        try:
            # Check critical components only
            critical_results = {}
            for name, check in self.health_checker._checks.items():
                if check.critical:
                    result = await self.health_checker.run_check(name)
                    if result:
                        critical_results[name] = result

            # Service is ready if all critical components are healthy
            ready = all(result.status.value == "healthy" for result in critical_results.values())

            return {
                "ready": ready,
                "timestamp": time.time(),
                "critical_checks": {
                    name: result.status.value for name, result in critical_results.items()
                },
            }

        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return {"ready": False, "timestamp": time.time(), "error": str(e)}

    async def liveness_check(self) -> Dict[str, Any]:
        """Liveness check - determines if service is alive and functioning."""
        try:
            # Basic liveness - service is alive if it can respond
            return {
                "alive": True,
                "timestamp": time.time(),
                "uptime": time.time() - self._startup_time,
            }

        except Exception as e:
            logger.error(f"Liveness check failed: {e}")
            return {"alive": False, "timestamp": time.time(), "error": str(e)}

    async def dependencies_check(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check health of external dependencies."""
        try:
            return await self.service_monitor.get_system_dependencies_status(config)

        except Exception as e:
            logger.error(f"Dependencies check failed: {e}")
            return {"overall_health": "unhealthy", "error": str(e), "timestamp": time.time()}

    async def metrics_endpoint(self) -> Dict[str, Any]:
        """Metrics endpoint for monitoring systems."""
        try:
            if not self.metrics_collector:
                return {"error": "Metrics collection not enabled", "timestamp": time.time()}

            summary = self.metrics_collector.get_summary()

            # Convert metrics to Prometheus-like format
            metrics_data = {}
            for name, metric in summary.items():
                metrics_data[name] = {
                    "type": metric.metric_type.value,
                    "value": metric.current_value,
                    "count": metric.count,
                    "tags": metric.tags,
                }

            return {"metrics": metrics_data, "timestamp": time.time()}

        except Exception as e:
            logger.error(f"Metrics endpoint failed: {e}")
            return {"error": str(e), "timestamp": time.time()}

    async def detailed_status(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive status endpoint with all health information."""
        try:
            # Get all health information
            health_summary = self.health_checker.get_health_summary()
            dependencies_status = await self.service_monitor.get_system_dependencies_status(config)

            # Get metrics if available
            metrics_summary = None
            if self.metrics_collector:
                try:
                    from .metrics_collector import PerformanceMonitor

                    if hasattr(self, "performance_monitor"):
                        metrics_summary = self.performance_monitor.get_system_overview()
                    else:
                        metrics_summary = {
                            "total_metrics": len(self.metrics_collector.get_summary())
                        }
                except Exception as e:
                    logger.warning(f"Could not get metrics summary: {e}")

            return {
                "overall_status": health_summary["overall_status"],
                "timestamp": time.time(),
                "uptime": time.time() - self._startup_time,
                "health": health_summary,
                "dependencies": dependencies_status,
                "metrics": metrics_summary,
            }

        except Exception as e:
            logger.error(f"Detailed status failed: {e}")
            return {"overall_status": "unhealthy", "timestamp": time.time(), "error": str(e)}

    # Web framework specific implementations

    def flask_routes(self, app, config: Dict[str, Any]):
        """Register Flask routes for health endpoints."""
        try:
            from flask import jsonify, request

            @app.route("/health")
            async def health():
                include_details = request.args.get("details", "false").lower() == "true"
                result = await self.health_check(include_details)
                status_code = 200 if result.get("status") == "healthy" else 503
                return jsonify(result), status_code

            @app.route("/health/ready")
            async def ready():
                result = await self.readiness_check()
                status_code = 200 if result.get("ready") else 503
                return jsonify(result), status_code

            @app.route("/health/live")
            async def live():
                result = await self.liveness_check()
                status_code = 200 if result.get("alive") else 503
                return jsonify(result), status_code

            @app.route("/health/dependencies")
            async def dependencies():
                result = await self.dependencies_check(config)
                status_code = 200 if result.get("overall_health") == "healthy" else 503
                return jsonify(result), status_code

            @app.route("/metrics")
            async def metrics():
                result = await self.metrics_endpoint()
                status_code = 200 if "error" not in result else 503
                return jsonify(result), status_code

            @app.route("/status")
            async def status():
                result = await self.detailed_status(config)
                status_code = 200 if result.get("overall_status") == "healthy" else 503
                return jsonify(result), status_code

        except ImportError:
            logger.warning("Flask not available, skipping Flask routes")

    def fastapi_routes(self, app, config: Dict[str, Any]):
        """Register FastAPI routes for health endpoints."""
        try:
            from fastapi import Query
            from fastapi.responses import JSONResponse

            @app.get("/health")
            async def health(details: bool = Query(False)):
                result = await self.health_check(details)
                status_code = 200 if result.get("status") == "healthy" else 503
                return JSONResponse(result, status_code=status_code)

            @app.get("/health/ready")
            async def ready():
                result = await self.readiness_check()
                status_code = 200 if result.get("ready") else 503
                return JSONResponse(result, status_code=status_code)

            @app.get("/health/live")
            async def live():
                result = await self.liveness_check()
                status_code = 200 if result.get("alive") else 503
                return JSONResponse(result, status_code=status_code)

            @app.get("/health/dependencies")
            async def dependencies():
                result = await self.dependencies_check(config)
                status_code = 200 if result.get("overall_health") == "healthy" else 503
                return JSONResponse(result, status_code=status_code)

            @app.get("/metrics")
            async def metrics():
                result = await self.metrics_endpoint()
                status_code = 200 if "error" not in result else 503
                return JSONResponse(result, status_code=status_code)

            @app.get("/status")
            async def status():
                result = await self.detailed_status(config)
                status_code = 200 if result.get("overall_status") == "healthy" else 503
                return JSONResponse(result, status_code=status_code)

        except ImportError:
            logger.warning("FastAPI not available, skipping FastAPI routes")

    def aiohttp_routes(self, app, config: Dict[str, Any]):
        """Register aiohttp routes for health endpoints."""
        try:
            from aiohttp import web

            async def health(request):
                include_details = request.query.get("details", "false").lower() == "true"
                result = await self.health_check(include_details)
                status = 200 if result.get("status") == "healthy" else 503
                return web.json_response(result, status=status)

            async def ready(request):
                result = await self.readiness_check()
                status = 200 if result.get("ready") else 503
                return web.json_response(result, status=status)

            async def live(request):
                result = await self.liveness_check()
                status = 200 if result.get("alive") else 503
                return web.json_response(result, status=status)

            async def dependencies(request):
                result = await self.dependencies_check(config)
                status = 200 if result.get("overall_health") == "healthy" else 503
                return web.json_response(result, status=status)

            async def metrics(request):
                result = await self.metrics_endpoint()
                status = 200 if "error" not in result else 503
                return web.json_response(result, status=status)

            async def status(request):
                result = await self.detailed_status(config)
                status = 200 if result.get("overall_status") == "healthy" else 503
                return web.json_response(result, status=status)

            # Add routes
            app.router.add_get("/health", health)
            app.router.add_get("/health/ready", ready)
            app.router.add_get("/health/live", live)
            app.router.add_get("/health/dependencies", dependencies)
            app.router.add_get("/metrics", metrics)
            app.router.add_get("/status", status)

        except ImportError:
            logger.warning("aiohttp not available, skipping aiohttp routes")

    def prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        try:
            if not self.metrics_collector:
                return "# Metrics collection not enabled\n"

            summary = self.metrics_collector.get_summary()
            prometheus_lines = []

            for name, metric in summary.items():
                # Convert metric name to Prometheus format
                prom_name = name.replace(".", "_").replace("-", "_")

                # Add help and type comments
                prometheus_lines.append(f"# HELP {prom_name} {metric.metric_type.value} metric")
                prometheus_lines.append(f"# TYPE {prom_name} {metric.metric_type.value}")

                # Add metric value with tags
                tags_str = ""
                if metric.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in metric.tags.items()]
                    tags_str = "{" + ",".join(tag_pairs) + "}"

                prometheus_lines.append(f"{prom_name}{tags_str} {metric.current_value}")

            return "\n".join(prometheus_lines) + "\n"

        except Exception as e:
            logger.error(f"Prometheus metrics export failed: {e}")
            return f"# Error exporting metrics: {e}\n"
