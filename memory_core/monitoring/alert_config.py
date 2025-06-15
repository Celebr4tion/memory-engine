"""
Alert configuration and management for Memory Engine monitoring.

This module defines alert rules, thresholds, and notification channels
for critical system issues and performance degradation.
"""

import logging
import time
import smtplib
import json
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

import httpx

from .structured_logger import get_logger
from .performance_monitor import PerformanceAlert


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class AlertChannel(Enum):
    """Alert notification channels."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    CONSOLE = "console"


@dataclass
class AlertRule:
    """Definition of an alert rule."""

    name: str
    metric_type: str
    condition: str  # e.g., "gt", "lt", "eq"
    threshold: float
    severity: AlertSeverity
    description: str
    enabled: bool = True
    cooldown_minutes: int = 5
    notification_channels: List[AlertChannel] = field(default_factory=list)
    custom_message: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertNotification:
    """Alert notification to be sent."""

    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_value: float
    threshold: float
    channels: List[AlertChannel]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationChannel:
    """Configuration for notification channels."""

    channel_type: AlertChannel
    config: Dict[str, Any]
    enabled: bool = True


class AlertManager:
    """
    Manages alert rules, evaluates conditions, and sends notifications.
    """

    def __init__(
        self,
        alert_rules: Optional[List[AlertRule]] = None,
        notification_channels: Optional[List[NotificationChannel]] = None,
    ):
        """
        Initialize alert manager.

        Args:
            alert_rules: List of alert rules to monitor
            notification_channels: Available notification channels
        """
        self.logger = get_logger(__name__, "alert_manager")

        # Initialize alert rules with defaults
        self.alert_rules = alert_rules or self._get_default_alert_rules()
        self.notification_channels = notification_channels or []

        # Track recent alerts to prevent spam
        self.recent_alerts: Dict[str, datetime] = {}

        # Alert history
        self.alert_history: List[AlertNotification] = []
        self.max_history_size = 1000

        self.logger.info(
            "Alert manager initialized",
            rules_count=len(self.alert_rules),
            channels_count=len(self.notification_channels),
        )

    def _get_default_alert_rules(self) -> List[AlertRule]:
        """Get default alert rules for Memory Engine."""
        return [
            # System Resource Alerts
            AlertRule(
                name="high_cpu_utilization",
                metric_type="cpu_percent",
                condition="gt",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                description="CPU utilization is above 85%",
                notification_channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
                tags={"category": "system", "component": "cpu"},
            ),
            AlertRule(
                name="critical_cpu_utilization",
                metric_type="cpu_percent",
                condition="gt",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                description="CPU utilization is critically high (>95%)",
                notification_channels=[
                    AlertChannel.EMAIL,
                    AlertChannel.WEBHOOK,
                    AlertChannel.CONSOLE,
                ],
                tags={"category": "system", "component": "cpu"},
            ),
            AlertRule(
                name="high_memory_utilization",
                metric_type="memory_percent",
                condition="gt",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                description="Memory utilization is above 80%",
                notification_channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
                tags={"category": "system", "component": "memory"},
            ),
            AlertRule(
                name="critical_memory_utilization",
                metric_type="memory_percent",
                condition="gt",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                description="Memory utilization is critically high (>95%)",
                notification_channels=[
                    AlertChannel.EMAIL,
                    AlertChannel.WEBHOOK,
                    AlertChannel.CONSOLE,
                ],
                tags={"category": "system", "component": "memory"},
            ),
            # Query Performance Alerts
            AlertRule(
                name="slow_query_performance",
                metric_type="query_avg_time_ms",
                condition="gt",
                threshold=5000.0,
                severity=AlertSeverity.WARNING,
                description="Average query execution time is above 5 seconds",
                notification_channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
                tags={"category": "performance", "component": "query"},
            ),
            AlertRule(
                name="very_slow_query_performance",
                metric_type="query_avg_time_ms",
                condition="gt",
                threshold=15000.0,
                severity=AlertSeverity.CRITICAL,
                description="Average query execution time is above 15 seconds",
                notification_channels=[
                    AlertChannel.EMAIL,
                    AlertChannel.WEBHOOK,
                    AlertChannel.CONSOLE,
                ],
                tags={"category": "performance", "component": "query"},
            ),
            AlertRule(
                name="high_query_error_rate",
                metric_type="query_error_rate",
                condition="gt",
                threshold=0.05,
                severity=AlertSeverity.WARNING,
                description="Query error rate is above 5%",
                notification_channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
                tags={"category": "reliability", "component": "query"},
            ),
            AlertRule(
                name="critical_query_error_rate",
                metric_type="query_error_rate",
                condition="gt",
                threshold=0.15,
                severity=AlertSeverity.CRITICAL,
                description="Query error rate is critically high (>15%)",
                notification_channels=[
                    AlertChannel.EMAIL,
                    AlertChannel.WEBHOOK,
                    AlertChannel.CONSOLE,
                ],
                tags={"category": "reliability", "component": "query"},
            ),
            # Cache Performance Alerts
            AlertRule(
                name="low_cache_hit_rate",
                metric_type="cache_hit_rate",
                condition="lt",
                threshold=0.3,
                severity=AlertSeverity.WARNING,
                description="Cache hit rate is below 30%",
                notification_channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
                tags={"category": "performance", "component": "cache"},
            ),
            # Ingestion Performance Alerts
            AlertRule(
                name="high_ingestion_error_rate",
                metric_type="ingestion_error_rate",
                condition="gt",
                threshold=0.02,
                severity=AlertSeverity.WARNING,
                description="Ingestion error rate is above 2%",
                notification_channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
                tags={"category": "reliability", "component": "ingestion"},
            ),
            AlertRule(
                name="critical_ingestion_error_rate",
                metric_type="ingestion_error_rate",
                condition="gt",
                threshold=0.10,
                severity=AlertSeverity.CRITICAL,
                description="Ingestion error rate is critically high (>10%)",
                notification_channels=[
                    AlertChannel.EMAIL,
                    AlertChannel.WEBHOOK,
                    AlertChannel.CONSOLE,
                ],
                tags={"category": "reliability", "component": "ingestion"},
            ),
            # Health Check Alerts
            AlertRule(
                name="janusgraph_unhealthy",
                metric_type="component_health",
                condition="eq",
                threshold=0,  # 0 = unhealthy
                severity=AlertSeverity.CRITICAL,
                description="JanusGraph database is unhealthy",
                notification_channels=[
                    AlertChannel.EMAIL,
                    AlertChannel.WEBHOOK,
                    AlertChannel.CONSOLE,
                ],
                tags={"category": "health", "component": "janusgraph"},
            ),
            AlertRule(
                name="milvus_unhealthy",
                metric_type="component_health",
                condition="eq",
                threshold=0,  # 0 = unhealthy
                severity=AlertSeverity.CRITICAL,
                description="Milvus vector database is unhealthy",
                notification_channels=[
                    AlertChannel.EMAIL,
                    AlertChannel.WEBHOOK,
                    AlertChannel.CONSOLE,
                ],
                tags={"category": "health", "component": "milvus"},
            ),
            AlertRule(
                name="gemini_api_unhealthy",
                metric_type="component_health",
                condition="eq",
                threshold=0,  # 0 = unhealthy
                severity=AlertSeverity.CRITICAL,
                description="Gemini API is unhealthy",
                notification_channels=[
                    AlertChannel.EMAIL,
                    AlertChannel.WEBHOOK,
                    AlertChannel.CONSOLE,
                ],
                tags={"category": "health", "component": "gemini_api"},
            ),
        ]

    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.alert_rules.append(rule)
        self.logger.info("Alert rule added", rule_name=rule.name, severity=rule.severity.value)

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule by name."""
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]
        self.logger.info("Alert rule removed", rule_name=rule_name)

    def add_notification_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.notification_channels.append(channel)
        self.logger.info(
            "Notification channel added",
            channel_type=channel.channel_type.value,
            enabled=channel.enabled,
        )

    def evaluate_metrics(self, metrics: Dict[str, float]) -> List[AlertNotification]:
        """
        Evaluate current metrics against alert rules.

        Args:
            metrics: Dictionary of metric name -> value

        Returns:
            List of alerts to send
        """
        triggered_alerts = []
        current_time = datetime.now(UTC)

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            metric_value = metrics.get(rule.metric_type)
            if metric_value is None:
                continue

            # Check if alert condition is met
            if self._evaluate_condition(metric_value, rule.condition, rule.threshold):
                # Check cooldown
                last_alert_time = self.recent_alerts.get(rule.name)
                if (
                    last_alert_time
                    and (current_time - last_alert_time).total_seconds()
                    < rule.cooldown_minutes * 60
                ):
                    continue

                # Create alert notification
                alert = AlertNotification(
                    alert_id=f"{rule.name}_{int(time.time())}",
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=rule.custom_message or rule.description,
                    timestamp=current_time,
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    channels=rule.notification_channels,
                    metadata={
                        "tags": rule.tags,
                        "condition": rule.condition,
                        "metric_type": rule.metric_type,
                    },
                )

                triggered_alerts.append(alert)
                self.recent_alerts[rule.name] = current_time

        return triggered_alerts

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate if a metric value meets the alert condition."""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        else:
            self.logger.warning("Unknown condition", condition=condition)
            return False

    async def send_alert(self, alert: AlertNotification):
        """
        Send an alert through configured notification channels.

        Args:
            alert: Alert notification to send
        """
        for channel_type in alert.channels:
            try:
                await self._send_to_channel(alert, channel_type)
                self.logger.info(
                    "Alert sent successfully",
                    alert_id=alert.alert_id,
                    channel=channel_type.value,
                    severity=alert.severity.value,
                )
            except Exception as e:
                self.logger.error(
                    "Failed to send alert",
                    alert_id=alert.alert_id,
                    channel=channel_type.value,
                    error=str(e),
                )

        # Add to history
        self.alert_history.append(alert)

        # Trim history if needed
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size :]

    async def _send_to_channel(self, alert: AlertNotification, channel_type: AlertChannel):
        """Send alert to specific channel type."""
        # Find channel configuration
        channel_config = None
        for channel in self.notification_channels:
            if channel.channel_type == channel_type and channel.enabled:
                channel_config = channel.config
                break

        if channel_type == AlertChannel.CONSOLE:
            await self._send_console_alert(alert)
        elif channel_type == AlertChannel.EMAIL and channel_config:
            await self._send_email_alert(alert, channel_config)
        elif channel_type == AlertChannel.WEBHOOK and channel_config:
            await self._send_webhook_alert(alert, channel_config)
        elif channel_type == AlertChannel.SLACK and channel_config:
            await self._send_slack_alert(alert, channel_config)
        else:
            self.logger.warning(
                "No configuration found for channel", channel_type=channel_type.value
            )

    async def _send_console_alert(self, alert: AlertNotification):
        """Send alert to console/logs."""
        log_level = (
            logging.CRITICAL if alert.severity == AlertSeverity.CRITICAL else logging.WARNING
        )
        self.logger.log(
            log_level,
            f"ALERT [{alert.severity.value.upper()}]: {alert.message}",
            alert_id=alert.alert_id,
            rule_name=alert.rule_name,
            metric_value=alert.metric_value,
            threshold=alert.threshold,
            **alert.metadata,
        )

    async def _send_email_alert(self, alert: AlertNotification, config: Dict[str, Any]):
        """Send alert via email."""
        try:
            msg = MimeMultipart()
            msg["From"] = config["from_email"]
            msg["To"] = ", ".join(config["to_emails"])
            msg["Subject"] = (
                f"[Memory Engine Alert] {alert.severity.value.upper()}: {alert.rule_name}"
            )

            body = self._format_email_body(alert)
            msg.attach(MimeText(body, "html"))

            # Send email
            with smtplib.SMTP(config["smtp_host"], config.get("smtp_port", 587)) as server:
                if config.get("use_tls", True):
                    server.starttls()
                if config.get("username") and config.get("password"):
                    server.login(config["username"], config["password"])
                server.send_message(msg)

        except Exception as e:
            self.logger.error("Failed to send email alert", alert_id=alert.alert_id, error=str(e))
            raise

    async def _send_webhook_alert(self, alert: AlertNotification, config: Dict[str, Any]):
        """Send alert via webhook."""
        payload = {
            "alert_id": alert.alert_id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "metric_value": alert.metric_value,
            "threshold": alert.threshold,
            "metadata": alert.metadata,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                config["url"], json=payload, headers=config.get("headers", {}), timeout=30.0
            )
            response.raise_for_status()

    async def _send_slack_alert(self, alert: AlertNotification, config: Dict[str, Any]):
        """Send alert to Slack."""
        webhook_url = config["webhook_url"]

        # Format Slack message
        color = self._get_slack_color(alert.severity)

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"Memory Engine Alert: {alert.rule_name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Metric Value", "value": str(alert.metric_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True,
                        },
                    ],
                    "footer": "Memory Engine Monitoring",
                    "ts": int(alert.timestamp.timestamp()),
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload, timeout=30.0)
            response.raise_for_status()

    def _get_slack_color(self, severity: AlertSeverity) -> str:
        """Get Slack attachment color for severity."""
        color_map = {
            AlertSeverity.LOW: "#36a64f",  # Green
            AlertSeverity.WARNING: "#ffb366",  # Orange
            AlertSeverity.CRITICAL: "#ff6b6b",  # Red
            AlertSeverity.FATAL: "#8b0000",  # Dark Red
        }
        return color_map.get(severity, "#808080")  # Gray default

    def _format_email_body(self, alert: AlertNotification) -> str:
        """Format email body for alerts."""
        return f"""
        <html>
        <body>
            <h2>Memory Engine Alert</h2>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr><td><strong>Rule Name:</strong></td><td>{alert.rule_name}</td></tr>
                <tr><td><strong>Severity:</strong></td><td><span style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange'}">{alert.severity.value.upper()}</span></td></tr>
                <tr><td><strong>Message:</strong></td><td>{alert.message}</td></tr>
                <tr><td><strong>Metric Value:</strong></td><td>{alert.metric_value}</td></tr>
                <tr><td><strong>Threshold:</strong></td><td>{alert.threshold}</td></tr>
                <tr><td><strong>Timestamp:</strong></td><td>{alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}</td></tr>
            </table>
            
            <h3>Additional Information</h3>
            <ul>
                <li><strong>Alert ID:</strong> {alert.alert_id}</li>
                <li><strong>Tags:</strong> {', '.join(f"{k}={v}" for k, v in alert.metadata.get('tags', {}).items())}</li>
            </ul>
            
            <p><em>This alert was generated by Memory Engine monitoring system.</em></p>
        </body>
        </html>
        """

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics and recent activity."""
        if not self.alert_history:
            return {
                "total_alerts": 0,
                "alerts_by_severity": {},
                "alerts_by_rule": {},
                "recent_alerts": [],
            }

        # Count by severity
        severity_counts = {}
        rule_counts = {}

        # Last 24 hours
        cutoff_time = datetime.now(UTC) - timedelta(days=1)
        recent_alerts = [alert for alert in self.alert_history if alert.timestamp > cutoff_time]

        for alert in recent_alerts:
            severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
            rule_counts[alert.rule_name] = rule_counts.get(alert.rule_name, 0) + 1

        return {
            "total_alerts": len(self.alert_history),
            "recent_alerts_24h": len(recent_alerts),
            "alerts_by_severity": severity_counts,
            "alerts_by_rule": rule_counts,
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_value": alert.metric_value,
                }
                for alert in self.alert_history[-10:]  # Last 10 alerts
            ],
        }


def create_default_alert_manager() -> AlertManager:
    """Create alert manager with default configuration."""
    return AlertManager()


def create_email_notification_channel(
    from_email: str,
    to_emails: List[str],
    smtp_host: str,
    smtp_port: int = 587,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_tls: bool = True,
) -> NotificationChannel:
    """Create email notification channel configuration."""
    return NotificationChannel(
        channel_type=AlertChannel.EMAIL,
        config={
            "from_email": from_email,
            "to_emails": to_emails,
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "use_tls": use_tls,
        },
    )


def create_webhook_notification_channel(
    url: str, headers: Optional[Dict[str, str]] = None
) -> NotificationChannel:
    """Create webhook notification channel configuration."""
    return NotificationChannel(
        channel_type=AlertChannel.WEBHOOK, config={"url": url, "headers": headers or {}}
    )


def create_slack_notification_channel(webhook_url: str) -> NotificationChannel:
    """Create Slack notification channel configuration."""
    return NotificationChannel(channel_type=AlertChannel.SLACK, config={"webhook_url": webhook_url})
