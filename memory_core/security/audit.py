"""
Audit logging system for the Memory Engine.

Provides comprehensive audit logging for security-sensitive operations,
compliance tracking, and forensic analysis.
"""

import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta, UTC
import uuid
import os
from pathlib import Path

from memory_core.config.config_manager import get_config


logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Audit severity levels."""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"


class AuditCategory(Enum):
    """Categories of auditable events."""
    
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    KNOWLEDGE_ACCESS = "knowledge_access"
    KNOWLEDGE_MODIFICATION = "knowledge_modification"
    USER_MANAGEMENT = "user_management"
    ROLE_MANAGEMENT = "role_management"
    PRIVACY_CONTROL = "privacy_control"
    SYSTEM_CONFIGURATION = "system_configuration"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    ENCRYPTION = "encryption"
    SECURITY_INCIDENT = "security_incident"


@dataclass
class AuditEvent:
    """Individual audit event record."""
    
    event_id: str
    timestamp: datetime
    level: AuditLevel
    category: AuditCategory
    action: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0  # 0.0 = low risk, 1.0 = high risk
    compliance_tags: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None  # For grouping related events
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary representation."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        data['category'] = self.category.value
        return data
    
    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    def get_signature(self) -> str:
        """Generate a cryptographic signature for the event."""
        event_data = self.to_json()
        return hashlib.sha256(event_data.encode('utf-8')).hexdigest()


@dataclass
class AuditFilter:
    """Filter criteria for audit log queries."""
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    levels: Optional[List[AuditLevel]] = None
    categories: Optional[List[AuditCategory]] = None
    user_ids: Optional[List[str]] = None
    actions: Optional[List[str]] = None
    resource_types: Optional[List[str]] = None
    success_only: Optional[bool] = None
    min_risk_score: Optional[float] = None
    compliance_tags: Optional[List[str]] = None
    ip_addresses: Optional[List[str]] = None
    limit: Optional[int] = None


class AuditLogger:
    """
    Comprehensive audit logging system for security and compliance.
    """
    
    def __init__(self, log_directory: Optional[str] = None):
        """
        Initialize the audit logger.
        
        Args:
            log_directory: Directory to store audit logs
        """
        self.config = get_config()
        
        # Set up log directory
        if log_directory:
            self.log_directory = Path(log_directory)
        else:
            self.log_directory = Path(self.config.get('audit.log_directory', 'logs/audit'))
        
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for recent events (configurable size)
        self.max_memory_events = self.config.get('audit.max_memory_events', 10000)
        self._memory_events: List[AuditEvent] = []
        
        # Security settings
        self.enable_signatures = self.config.get('audit.enable_signatures', True)
        self.enable_encryption = self.config.get('audit.enable_encryption', False)
        
        # Compliance settings
        self.retention_days = self.config.get('audit.retention_days', 2555)  # 7 years default
        self.auto_archive = self.config.get('audit.auto_archive', True)
        
        # Set up file logging
        self._setup_file_logging()
        
        # Event correlations for detecting patterns
        self._correlations: Dict[str, List[str]] = {}
        
        logger.info(f"AuditLogger initialized with log directory: {self.log_directory}")
    
    def _setup_file_logging(self) -> None:
        """Set up structured file logging for audit events."""
        # Create a separate logger for audit events
        self.audit_file_logger = logging.getLogger('memory_engine.audit')
        self.audit_file_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.audit_file_logger.handlers[:]:
            self.audit_file_logger.removeHandler(handler)
        
        # Create file handler with daily rotation
        from logging.handlers import TimedRotatingFileHandler
        
        audit_file = self.log_directory / "audit.log"
        file_handler = TimedRotatingFileHandler(
            audit_file,
            when='midnight',
            interval=1,
            backupCount=self.retention_days,
            encoding='utf-8'
        )
        
        # Use JSON formatter for structured logging
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        
        self.audit_file_logger.addHandler(file_handler)
        self.audit_file_logger.propagate = False
    
    def log_event(
        self,
        level: AuditLevel,
        category: AuditCategory,
        action: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_score: float = 0.0,
        compliance_tags: Optional[List[str]] = None,
        correlation_id: Optional[str] = None
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            level: Audit severity level
            category: Event category
            action: Action being audited
            user_id: User performing the action
            session_id: Session identifier
            resource_type: Type of resource being accessed
            resource_id: Resource identifier
            ip_address: Client IP address
            user_agent: Client user agent
            success: Whether the action succeeded
            error_message: Error message if action failed
            details: Additional event details
            risk_score: Risk assessment score (0.0-1.0)
            compliance_tags: Compliance-related tags
            correlation_id: ID for correlating related events
        
        Returns:
            Created AuditEvent object
        """
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            level=level,
            category=category,
            action=action,
            user_id=user_id,
            session_id=session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            details=details or {},
            risk_score=risk_score,
            compliance_tags=compliance_tags or [],
            correlation_id=correlation_id
        )
        
        # Store in memory
        self._memory_events.append(event)
        
        # Maintain memory limit
        if len(self._memory_events) > self.max_memory_events:
            self._memory_events.pop(0)
        
        # Log to file
        self.audit_file_logger.info(event.to_json())
        
        # Update correlations
        if correlation_id:
            if correlation_id not in self._correlations:
                self._correlations[correlation_id] = []
            self._correlations[correlation_id].append(event.event_id)
        
        # Check for security patterns
        self._analyze_security_patterns(event)
        
        return event
    
    def _analyze_security_patterns(self, event: AuditEvent) -> None:
        """Analyze events for suspicious patterns."""
        
        # Multiple failed authentication attempts
        if (event.category == AuditCategory.AUTHENTICATION and 
            not event.success and event.user_id):
            
            recent_failures = self.query_events(
                AuditFilter(
                    start_time=datetime.now(UTC) - timedelta(minutes=15),
                    categories=[AuditCategory.AUTHENTICATION],
                    user_ids=[event.user_id],
                    success_only=False
                )
            )
            
            failed_attempts = [e for e in recent_failures if not e.success]
            
            if len(failed_attempts) >= 5:
                self.log_event(
                    AuditLevel.SECURITY,
                    AuditCategory.SECURITY_INCIDENT,
                    "Multiple failed authentication attempts detected",
                    user_id=event.user_id,
                    ip_address=event.ip_address,
                    risk_score=0.8,
                    details={
                        'failed_attempts_count': len(failed_attempts),
                        'time_window_minutes': 15,
                        'trigger_event_id': event.event_id
                    },
                    compliance_tags=['security_alert', 'brute_force_detection']
                )
        
        # Unusual access patterns
        if event.category == AuditCategory.KNOWLEDGE_ACCESS and event.user_id:
            # Check for access from new IP addresses
            recent_access = self.query_events(
                AuditFilter(
                    start_time=datetime.now(UTC) - timedelta(days=30),
                    categories=[AuditCategory.KNOWLEDGE_ACCESS],
                    user_ids=[event.user_id]
                )
            )
            
            known_ips = {e.ip_address for e in recent_access if e.ip_address}
            
            if event.ip_address and event.ip_address not in known_ips and len(known_ips) > 0:
                self.log_event(
                    AuditLevel.WARNING,
                    AuditCategory.SECURITY_INCIDENT,
                    "Access from new IP address",
                    user_id=event.user_id,
                    ip_address=event.ip_address,
                    risk_score=0.4,
                    details={
                        'new_ip': event.ip_address,
                        'known_ips': list(known_ips),
                        'trigger_event_id': event.event_id
                    },
                    compliance_tags=['access_anomaly']
                )
    
    def log_authentication(
        self,
        action: str,
        user_id: Optional[str] = None,
        success: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log authentication events."""
        risk_score = 0.2 if success else 0.6
        
        return self.log_event(
            AuditLevel.INFO if success else AuditLevel.WARNING,
            AuditCategory.AUTHENTICATION,
            action,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            details=details,
            risk_score=risk_score,
            compliance_tags=['authentication', 'access_control']
        )
    
    def log_authorization(
        self,
        action: str,
        user_id: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log authorization events."""
        risk_score = 0.1 if success else 0.7
        
        return self.log_event(
            AuditLevel.INFO if success else AuditLevel.WARNING,
            AuditCategory.AUTHORIZATION,
            action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            error_message=error_message,
            details=details,
            risk_score=risk_score,
            compliance_tags=['authorization', 'access_control']
        )
    
    def log_knowledge_access(
        self,
        action: str,
        user_id: str,
        resource_id: str,
        resource_type: str = "knowledge_node",
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log knowledge access events."""
        return self.log_event(
            AuditLevel.INFO,
            AuditCategory.KNOWLEDGE_ACCESS,
            action,
            user_id=user_id,
            session_id=session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            risk_score=0.1,
            compliance_tags=['data_access', 'knowledge_management']
        )
    
    def log_knowledge_modification(
        self,
        action: str,
        user_id: str,
        resource_id: str,
        resource_type: str = "knowledge_node",
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log knowledge modification events."""
        risk_score = 0.3 if action in ['create', 'update'] else 0.5  # Higher risk for delete
        
        return self.log_event(
            AuditLevel.INFO,
            AuditCategory.KNOWLEDGE_MODIFICATION,
            action,
            user_id=user_id,
            session_id=session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            risk_score=risk_score,
            compliance_tags=['data_modification', 'knowledge_management']
        )
    
    def log_user_management(
        self,
        action: str,
        admin_user_id: str,
        target_user_id: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log user management events."""
        risk_score = 0.4 if action in ['create', 'update'] else 0.6  # Higher risk for delete
        
        return self.log_event(
            AuditLevel.INFO if success else AuditLevel.WARNING,
            AuditCategory.USER_MANAGEMENT,
            action,
            user_id=admin_user_id,
            resource_type="user",
            resource_id=target_user_id,
            success=success,
            details=details,
            risk_score=risk_score,
            compliance_tags=['user_management', 'administrative_action']
        )
    
    def log_privacy_control(
        self,
        action: str,
        user_id: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log privacy control events."""
        return self.log_event(
            AuditLevel.INFO,
            AuditCategory.PRIVACY_CONTROL,
            action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            risk_score=0.3,
            compliance_tags=['privacy', 'access_control', 'data_governance']
        )
    
    def log_security_incident(
        self,
        action: str,
        level: AuditLevel = AuditLevel.SECURITY,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        risk_score: float = 0.8,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log security incidents."""
        return self.log_event(
            level,
            AuditCategory.SECURITY_INCIDENT,
            action,
            user_id=user_id,
            ip_address=ip_address,
            risk_score=risk_score,
            details=details,
            compliance_tags=['security_incident', 'threat_detection']
        )
    
    def query_events(self, filter_criteria: AuditFilter) -> List[AuditEvent]:
        """
        Query audit events based on filter criteria.
        
        Args:
            filter_criteria: Filter criteria for the query
        
        Returns:
            List of matching audit events
        """
        events = self._memory_events.copy()
        
        # Apply time filters
        if filter_criteria.start_time:
            events = [e for e in events if e.timestamp >= filter_criteria.start_time]
        
        if filter_criteria.end_time:
            events = [e for e in events if e.timestamp <= filter_criteria.end_time]
        
        # Apply level filters
        if filter_criteria.levels:
            events = [e for e in events if e.level in filter_criteria.levels]
        
        # Apply category filters
        if filter_criteria.categories:
            events = [e for e in events if e.category in filter_criteria.categories]
        
        # Apply user filters
        if filter_criteria.user_ids:
            events = [e for e in events if e.user_id in filter_criteria.user_ids]
        
        # Apply action filters
        if filter_criteria.actions:
            events = [e for e in events if e.action in filter_criteria.actions]
        
        # Apply resource type filters
        if filter_criteria.resource_types:
            events = [e for e in events if e.resource_type in filter_criteria.resource_types]
        
        # Apply success filter
        if filter_criteria.success_only is not None:
            events = [e for e in events if e.success == filter_criteria.success_only]
        
        # Apply risk score filter
        if filter_criteria.min_risk_score is not None:
            events = [e for e in events if e.risk_score >= filter_criteria.min_risk_score]
        
        # Apply compliance tag filters
        if filter_criteria.compliance_tags:
            events = [
                e for e in events 
                if any(tag in e.compliance_tags for tag in filter_criteria.compliance_tags)
            ]
        
        # Apply IP address filters
        if filter_criteria.ip_addresses:
            events = [e for e in events if e.ip_address in filter_criteria.ip_addresses]
        
        # Sort by timestamp (most recent first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if filter_criteria.limit:
            events = events[:filter_criteria.limit]
        
        return events
    
    def get_security_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get a security summary for the specified time period.
        
        Args:
            days_back: Number of days to analyze
        
        Returns:
            Security summary statistics
        """
        start_time = datetime.now(UTC) - timedelta(days=days_back)
        
        all_events = self.query_events(AuditFilter(start_time=start_time))
        
        summary = {
            'period_days': days_back,
            'total_events': len(all_events),
            'events_by_level': {},
            'events_by_category': {},
            'failed_authentications': 0,
            'unauthorized_access_attempts': 0,
            'high_risk_events': 0,
            'unique_users': set(),
            'unique_ips': set(),
            'security_incidents': 0
        }
        
        for event in all_events:
            # Count by level
            level = event.level.value
            summary['events_by_level'][level] = summary['events_by_level'].get(level, 0) + 1
            
            # Count by category
            category = event.category.value
            summary['events_by_category'][category] = summary['events_by_category'].get(category, 0) + 1
            
            # Track specific security metrics
            if event.category == AuditCategory.AUTHENTICATION and not event.success:
                summary['failed_authentications'] += 1
            
            if event.category == AuditCategory.AUTHORIZATION and not event.success:
                summary['unauthorized_access_attempts'] += 1
            
            if event.risk_score >= 0.7:
                summary['high_risk_events'] += 1
            
            if event.category == AuditCategory.SECURITY_INCIDENT:
                summary['security_incidents'] += 1
            
            # Track users and IPs
            if event.user_id:
                summary['unique_users'].add(event.user_id)
            
            if event.ip_address:
                summary['unique_ips'].add(event.ip_address)
        
        # Convert sets to counts
        summary['unique_users'] = len(summary['unique_users'])
        summary['unique_ips'] = len(summary['unique_ips'])
        
        return summary
    
    def get_compliance_report(self, tags: List[str], days_back: int = 30) -> Dict[str, Any]:
        """
        Generate a compliance report for specific tags.
        
        Args:
            tags: Compliance tags to include in the report
            days_back: Number of days to analyze
        
        Returns:
            Compliance report
        """
        start_time = datetime.now(UTC) - timedelta(days=days_back)
        
        events = self.query_events(
            AuditFilter(
                start_time=start_time,
                compliance_tags=tags
            )
        )
        
        report = {
            'period_days': days_back,
            'compliance_tags': tags,
            'total_events': len(events),
            'events_by_tag': {},
            'events_by_user': {},
            'events_by_action': {},
            'risk_distribution': {
                'low': 0,      # 0.0 - 0.3
                'medium': 0,   # 0.3 - 0.7
                'high': 0      # 0.7 - 1.0
            }
        }
        
        for event in events:
            # Count by compliance tag
            for tag in event.compliance_tags:
                if tag in tags:
                    report['events_by_tag'][tag] = report['events_by_tag'].get(tag, 0) + 1
            
            # Count by user
            if event.user_id:
                report['events_by_user'][event.user_id] = report['events_by_user'].get(event.user_id, 0) + 1
            
            # Count by action
            report['events_by_action'][event.action] = report['events_by_action'].get(event.action, 0) + 1
            
            # Risk distribution
            if event.risk_score < 0.3:
                report['risk_distribution']['low'] += 1
            elif event.risk_score < 0.7:
                report['risk_distribution']['medium'] += 1
            else:
                report['risk_distribution']['high'] += 1
        
        return report
    
    def export_audit_log(
        self,
        output_file: str,
        filter_criteria: Optional[AuditFilter] = None,
        format_type: str = 'json'
    ) -> bool:
        """
        Export audit logs to a file.
        
        Args:
            output_file: Path to output file
            filter_criteria: Optional filter criteria
            format_type: Export format ('json' or 'csv')
        
        Returns:
            True if export successful, False otherwise
        """
        try:
            events = self.query_events(filter_criteria or AuditFilter())
            
            if format_type.lower() == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    event_dicts = [event.to_dict() for event in events]
                    json.dump(event_dicts, f, indent=2, sort_keys=True)
            
            elif format_type.lower() == 'csv':
                import csv
                
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    if events:
                        fieldnames = events[0].to_dict().keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for event in events:
                            writer.writerow(event.to_dict())
            
            else:
                logger.error(f"Unsupported export format: {format_type}")
                return False
            
            logger.info(f"Exported {len(events)} audit events to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export audit log: {str(e)}")
            return False
    
    def cleanup_old_events(self, retention_days: Optional[int] = None) -> int:
        """
        Clean up old audit events from memory.
        
        Args:
            retention_days: Number of days to retain (uses config default if None)
        
        Returns:
            Number of events cleaned up
        """
        retention_days = retention_days or self.retention_days
        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)
        
        initial_count = len(self._memory_events)
        self._memory_events = [
            event for event in self._memory_events
            if event.timestamp >= cutoff_date
        ]
        
        cleaned_count = initial_count - len(self._memory_events)
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old audit events")
        
        return cleaned_count