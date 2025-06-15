"""
Performance regression testing framework for Memory Engine.

This module provides automated testing capabilities to detect performance
regressions across different versions and configurations of the Memory Engine.
"""

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

from memory_core.monitoring.performance_monitor import PerformanceMonitor


@dataclass
class PerformanceBenchmark:
    """Represents a performance benchmark test."""

    name: str
    test_function: Callable
    description: str
    category: str  # 'query', 'ingestion', 'traversal', 'embedding'
    expected_max_time_ms: float
    expected_min_throughput: Optional[float] = None
    timeout_seconds: float = 300.0
    warmup_runs: int = 3
    test_runs: int = 10

    def __post_init__(self):
        self.id = f"{self.category}_{self.name}_{uuid.uuid4().hex[:8]}"


@dataclass
class BenchmarkResult:
    """Results from executing a performance benchmark."""

    benchmark_id: str
    benchmark_name: str
    category: str
    timestamp: float
    execution_times_ms: List[float]
    throughput_values: List[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def average_time_ms(self) -> float:
        """Average execution time."""
        return statistics.mean(self.execution_times_ms) if self.execution_times_ms else 0.0

    @property
    def median_time_ms(self) -> float:
        """Median execution time."""
        return statistics.median(self.execution_times_ms) if self.execution_times_ms else 0.0

    @property
    def p95_time_ms(self) -> float:
        """95th percentile execution time."""
        if len(self.execution_times_ms) >= 10:
            return statistics.quantiles(self.execution_times_ms, n=20)[18]
        return max(self.execution_times_ms) if self.execution_times_ms else 0.0

    @property
    def average_throughput(self) -> float:
        """Average throughput."""
        return statistics.mean(self.throughput_values) if self.throughput_values else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate of benchmark runs."""
        total_runs = self.success_count + self.error_count
        return self.success_count / total_runs if total_runs > 0 else 0.0


@dataclass
class RegressionReport:
    """Report comparing current performance against baseline."""

    benchmark_name: str
    category: str
    baseline_result: BenchmarkResult
    current_result: BenchmarkResult
    time_regression_percent: float
    throughput_regression_percent: float
    is_regression: bool
    severity: str  # 'none', 'minor', 'major', 'critical'

    def __post_init__(self):
        self.timestamp = time.time()


class PerformanceRegressionTester:
    """
    Automated performance regression testing framework.

    Features:
    - Configurable benchmark suites
    - Baseline performance recording
    - Regression detection and reporting
    - Historical performance tracking
    - Automated CI/CD integration
    - Detailed performance analysis
    """

    def __init__(
        self,
        baseline_storage_path: str = "performance_baselines.json",
        regression_threshold: float = 0.10,
    ):  # 10% regression threshold
        """
        Initialize the performance regression tester.

        Args:
            baseline_storage_path: Path to store baseline performance data
            regression_threshold: Threshold for detecting regressions (as percentage)
        """
        self.logger = logging.getLogger(__name__)
        self.baseline_storage_path = Path(baseline_storage_path)
        self.regression_threshold = regression_threshold

        # Registered benchmarks
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}

        # Test results storage
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        self.current_results: Dict[str, BenchmarkResult] = {}

        # Load existing baselines
        self._load_baselines()

        self.logger.info(
            f"Performance regression tester initialized with {len(self.baseline_results)} baselines"
        )

    def register_benchmark(self, benchmark: PerformanceBenchmark):
        """Register a performance benchmark for testing."""
        self.benchmarks[benchmark.name] = benchmark
        self.logger.info(f"Registered benchmark: {benchmark.name} ({benchmark.category})")

    def create_query_benchmark(
        self,
        name: str,
        query_function: Callable,
        expected_max_time_ms: float = 1000.0,
        description: str = "",
    ) -> PerformanceBenchmark:
        """Create and register a query performance benchmark."""
        benchmark = PerformanceBenchmark(
            name=name,
            test_function=query_function,
            description=description,
            category="query",
            expected_max_time_ms=expected_max_time_ms,
        )
        self.register_benchmark(benchmark)
        return benchmark

    def create_ingestion_benchmark(
        self,
        name: str,
        ingestion_function: Callable,
        expected_max_time_ms: float = 5000.0,
        expected_min_throughput: float = 10.0,
        description: str = "",
    ) -> PerformanceBenchmark:
        """Create and register an ingestion performance benchmark."""
        benchmark = PerformanceBenchmark(
            name=name,
            test_function=ingestion_function,
            description=description,
            category="ingestion",
            expected_max_time_ms=expected_max_time_ms,
            expected_min_throughput=expected_min_throughput,
        )
        self.register_benchmark(benchmark)
        return benchmark

    async def run_benchmark_suite(
        self, categories: Optional[List[str]] = None, benchmark_names: Optional[List[str]] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Run a suite of performance benchmarks.

        Args:
            categories: List of benchmark categories to run (None for all)
            benchmark_names: Specific benchmark names to run (None for all)

        Returns:
            Dictionary of benchmark results
        """
        # Filter benchmarks to run
        benchmarks_to_run = {}
        for name, benchmark in self.benchmarks.items():
            if categories and benchmark.category not in categories:
                continue
            if benchmark_names and name not in benchmark_names:
                continue
            benchmarks_to_run[name] = benchmark

        if not benchmarks_to_run:
            self.logger.warning("No benchmarks match the specified criteria")
            return {}

        self.logger.info(f"Running {len(benchmarks_to_run)} performance benchmarks")

        results = {}
        for name, benchmark in benchmarks_to_run.items():
            self.logger.info(f"Running benchmark: {name}")
            result = await self._run_single_benchmark(benchmark)
            results[name] = result
            self.current_results[name] = result

        return results

    async def _run_single_benchmark(self, benchmark: PerformanceBenchmark) -> BenchmarkResult:
        """Run a single performance benchmark."""
        result = BenchmarkResult(
            benchmark_id=benchmark.id,
            benchmark_name=benchmark.name,
            category=benchmark.category,
            timestamp=time.time(),
            execution_times_ms=[],
        )

        try:
            # Warmup runs
            self.logger.debug(
                f"Running {benchmark.warmup_runs} warmup iterations for {benchmark.name}"
            )
            for _ in range(benchmark.warmup_runs):
                try:
                    await asyncio.wait_for(
                        self._execute_benchmark_function(benchmark.test_function),
                        timeout=benchmark.timeout_seconds,
                    )
                except Exception as e:
                    self.logger.debug(f"Warmup run failed: {e}")

            # Actual test runs
            self.logger.debug(f"Running {benchmark.test_runs} test iterations for {benchmark.name}")
            for run_num in range(benchmark.test_runs):
                try:
                    start_time = time.time()

                    test_result = await asyncio.wait_for(
                        self._execute_benchmark_function(benchmark.test_function),
                        timeout=benchmark.timeout_seconds,
                    )

                    execution_time = (time.time() - start_time) * 1000  # Convert to ms
                    result.execution_times_ms.append(execution_time)
                    result.success_count += 1

                    # Extract throughput if available
                    if isinstance(test_result, dict) and "throughput" in test_result:
                        result.throughput_values.append(test_result["throughput"])
                    elif benchmark.expected_min_throughput and execution_time > 0:
                        # Calculate throughput as operations per second
                        throughput = 1000 / execution_time  # ops/sec
                        result.throughput_values.append(throughput)

                except asyncio.TimeoutError:
                    error_msg = f"Benchmark timeout after {benchmark.timeout_seconds}s"
                    result.errors.append(error_msg)
                    result.error_count += 1
                    self.logger.warning(f"Benchmark {benchmark.name} run {run_num + 1} timed out")

                except Exception as e:
                    error_msg = f"Benchmark execution error: {str(e)}"
                    result.errors.append(error_msg)
                    result.error_count += 1
                    self.logger.error(f"Benchmark {benchmark.name} run {run_num + 1} failed: {e}")

        except Exception as e:
            self.logger.error(f"Benchmark {benchmark.name} failed catastrophically: {e}")
            result.errors.append(f"Catastrophic failure: {str(e)}")
            result.error_count += 1

        # Log results summary
        if result.execution_times_ms:
            self.logger.info(
                f"Benchmark {benchmark.name} completed: "
                f"avg={result.average_time_ms:.2f}ms, "
                f"p95={result.p95_time_ms:.2f}ms, "
                f"success_rate={result.success_rate:.2%}"
            )
        else:
            self.logger.error(f"Benchmark {benchmark.name} produced no successful results")

        return result

    async def _execute_benchmark_function(self, test_function: Callable) -> Any:
        """Execute a benchmark test function."""
        if asyncio.iscoroutinefunction(test_function):
            return await test_function()
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, test_function)

    def detect_regressions(
        self, current_results: Optional[Dict[str, BenchmarkResult]] = None
    ) -> List[RegressionReport]:
        """
        Detect performance regressions by comparing against baselines.

        Args:
            current_results: Current benchmark results (uses self.current_results if None)

        Returns:
            List of regression reports
        """
        if current_results is None:
            current_results = self.current_results

        regression_reports = []

        for benchmark_name, current_result in current_results.items():
            baseline_result = self.baseline_results.get(benchmark_name)

            if not baseline_result:
                self.logger.warning(f"No baseline found for benchmark: {benchmark_name}")
                continue

            # Calculate regression percentages
            time_regression = 0.0
            if baseline_result.average_time_ms > 0:
                time_regression = (
                    current_result.average_time_ms - baseline_result.average_time_ms
                ) / baseline_result.average_time_ms

            throughput_regression = 0.0
            if baseline_result.average_throughput > 0 and current_result.average_throughput > 0:
                throughput_regression = (
                    baseline_result.average_throughput - current_result.average_throughput
                ) / baseline_result.average_throughput

            # Determine if this is a regression
            is_regression = (
                time_regression > self.regression_threshold
                or throughput_regression > self.regression_threshold
            )

            # Determine severity
            severity = "none"
            if is_regression:
                max_regression = max(time_regression, throughput_regression)
                if max_regression > 0.5:  # 50%
                    severity = "critical"
                elif max_regression > 0.25:  # 25%
                    severity = "major"
                else:
                    severity = "minor"

            report = RegressionReport(
                benchmark_name=benchmark_name,
                category=current_result.category,
                baseline_result=baseline_result,
                current_result=current_result,
                time_regression_percent=time_regression * 100,
                throughput_regression_percent=throughput_regression * 100,
                is_regression=is_regression,
                severity=severity,
            )

            regression_reports.append(report)

            if is_regression:
                self.logger.warning(
                    f"Performance regression detected in {benchmark_name}: "
                    f"time +{time_regression*100:.1f}%, throughput {throughput_regression*100:.1f}% "
                    f"(severity: {severity})"
                )

        return regression_reports

    def update_baselines(self, results: Optional[Dict[str, BenchmarkResult]] = None):
        """
        Update baseline performance metrics.

        Args:
            results: Results to use as new baselines (uses self.current_results if None)
        """
        if results is None:
            results = self.current_results

        updated_count = 0
        for name, result in results.items():
            if result.success_rate >= 0.8:  # Only update if at least 80% success rate
                self.baseline_results[name] = result
                updated_count += 1
            else:
                self.logger.warning(f"Skipping baseline update for {name} due to low success rate")

        # Save to disk
        self._save_baselines()

        self.logger.info(f"Updated {updated_count} performance baselines")

    def _load_baselines(self):
        """Load baseline performance data from disk."""
        if not self.baseline_storage_path.exists():
            self.logger.info("No baseline file found - starting fresh")
            return

        try:
            with open(self.baseline_storage_path, "r") as f:
                data = json.load(f)

            for name, result_data in data.items():
                # Reconstruct BenchmarkResult from JSON data
                result = BenchmarkResult(
                    benchmark_id=result_data["benchmark_id"],
                    benchmark_name=result_data["benchmark_name"],
                    category=result_data["category"],
                    timestamp=result_data["timestamp"],
                    execution_times_ms=result_data["execution_times_ms"],
                    throughput_values=result_data.get("throughput_values", []),
                    success_count=result_data["success_count"],
                    error_count=result_data["error_count"],
                    errors=result_data["errors"],
                )
                self.baseline_results[name] = result

            self.logger.info(f"Loaded {len(self.baseline_results)} baseline results")

        except Exception as e:
            self.logger.error(f"Failed to load baselines: {e}")

    def _save_baselines(self):
        """Save baseline performance data to disk."""
        try:
            # Convert results to JSON-serializable format
            data = {}
            for name, result in self.baseline_results.items():
                data[name] = {
                    "benchmark_id": result.benchmark_id,
                    "benchmark_name": result.benchmark_name,
                    "category": result.category,
                    "timestamp": result.timestamp,
                    "execution_times_ms": result.execution_times_ms,
                    "throughput_values": result.throughput_values,
                    "success_count": result.success_count,
                    "error_count": result.error_count,
                    "errors": result.errors,
                }

            with open(self.baseline_storage_path, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.debug(f"Saved baselines to {self.baseline_storage_path}")

        except Exception as e:
            self.logger.error(f"Failed to save baselines: {e}")

    def generate_regression_report(
        self, regression_reports: List[RegressionReport], format: str = "text"
    ) -> str:
        """
        Generate a human-readable regression report.

        Args:
            regression_reports: List of regression reports
            format: Report format ('text' or 'html')

        Returns:
            Formatted report string
        """
        if format == "text":
            return self._generate_text_report(regression_reports)
        elif format == "html":
            return self._generate_html_report(regression_reports)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _generate_text_report(self, reports: List[RegressionReport]) -> str:
        """Generate a text-based regression report."""
        lines = [
            "Performance Regression Report",
            "=" * 50,
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total benchmarks: {len(reports)}",
            "",
        ]

        # Summary
        regressions = [r for r in reports if r.is_regression]
        if regressions:
            lines.extend([f"⚠️  REGRESSIONS DETECTED: {len(regressions)}", ""])

            # Group by severity
            by_severity = {}
            for r in regressions:
                by_severity.setdefault(r.severity, []).append(r)

            for severity in ["critical", "major", "minor"]:
                if severity in by_severity:
                    lines.append(f"{severity.upper()}: {len(by_severity[severity])}")

            lines.append("")
        else:
            lines.extend(["✅ No regressions detected", ""])

        # Detailed results
        lines.append("Detailed Results:")
        lines.append("-" * 30)

        for report in sorted(reports, key=lambda x: x.time_regression_percent, reverse=True):
            status = "❌ REGRESSION" if report.is_regression else "✅ OK"

            lines.extend(
                [
                    f"{status} {report.benchmark_name} ({report.category})",
                    f"  Time: {report.baseline_result.average_time_ms:.2f}ms → "
                    f"{report.current_result.average_time_ms:.2f}ms "
                    f"({report.time_regression_percent:+.1f}%)",
                ]
            )

            if report.baseline_result.average_throughput > 0:
                lines.append(
                    f"  Throughput: {report.baseline_result.average_throughput:.2f} → "
                    f"{report.current_result.average_throughput:.2f} "
                    f"({report.throughput_regression_percent:+.1f}%)"
                )

            if report.is_regression:
                lines.append(f"  Severity: {report.severity}")

            lines.append("")

        return "\n".join(lines)

    def _generate_html_report(self, reports: List[RegressionReport]) -> str:
        """Generate an HTML-based regression report."""
        # This is a simplified HTML report - could be enhanced with CSS/charts
        regressions = [r for r in reports if r.is_regression]

        html = f"""
        <html>
        <head><title>Performance Regression Report</title></head>
        <body>
        <h1>Performance Regression Report</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Benchmarks: {len(reports)} | Regressions: {len(regressions)}</p>
        
        <h2>Summary</h2>
        """

        if regressions:
            html += f"<p style='color: red;'>⚠️ {len(regressions)} regressions detected</p>"
        else:
            html += "<p style='color: green;'>✅ No regressions detected</p>"

        html += "<h2>Results</h2><table border='1'>"
        html += "<tr><th>Benchmark</th><th>Category</th><th>Status</th><th>Time Change</th><th>Throughput Change</th></tr>"

        for report in reports:
            status_color = "red" if report.is_regression else "green"
            status_text = "REGRESSION" if report.is_regression else "OK"

            html += f"""
            <tr>
                <td>{report.benchmark_name}</td>
                <td>{report.category}</td>
                <td style='color: {status_color};'>{status_text}</td>
                <td>{report.time_regression_percent:+.1f}%</td>
                <td>{report.throughput_regression_percent:+.1f}%</td>
            </tr>
            """

        html += "</table></body></html>"
        return html


# Pre-defined benchmark functions for common operations
class CommonBenchmarks:
    """Collection of common benchmark functions."""

    @staticmethod
    def query_simple_search(query_engine, query: str = "test query"):
        """Simple search query benchmark."""
        from memory_core.query.query_types import QueryRequest, QueryType

        request = QueryRequest(query=query, query_type=QueryType.SEMANTIC_SEARCH, limit=10)

        start_time = time.time()
        response = query_engine.query(request)
        execution_time = time.time() - start_time

        return {
            "result_count": len(response.results),
            "execution_time_ms": execution_time * 1000,
            "cache_hit": response.from_cache,
        }

    @staticmethod
    def embedding_generation(embedding_manager, texts: List[str] = None):
        """Embedding generation benchmark."""
        if texts is None:
            texts = [f"Sample text for embedding {i}" for i in range(10)]

        start_time = time.time()
        embeddings = embedding_manager.generate_embeddings(texts)
        execution_time = time.time() - start_time

        return {
            "embeddings_count": len(embeddings),
            "execution_time_ms": execution_time * 1000,
            "throughput": len(embeddings) / execution_time if execution_time > 0 else 0,
        }

    @staticmethod
    def bulk_ingestion(bulk_processor, document_count: int = 100):
        """Bulk ingestion benchmark."""
        from memory_core.ingestion.bulk_processor import BulkDocument

        documents = [
            BulkDocument(
                id=f"test_doc_{i}",
                content=f"This is test document {i} with sample content for ingestion benchmarking.",
                source_label="benchmark",
            )
            for i in range(document_count)
        ]

        start_time = time.time()
        metrics = bulk_processor.process_documents(documents)
        execution_time = time.time() - start_time

        return {
            "documents_processed": metrics.processed_documents,
            "execution_time_ms": execution_time * 1000,
            "throughput": metrics.throughput_docs_per_second,
        }
