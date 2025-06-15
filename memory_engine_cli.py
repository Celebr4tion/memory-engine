#!/usr/bin/env python3
"""
Memory Engine CLI - Command line interface for Memory Engine management.

Usage:
    memory-engine init [--backend=BACKEND] [--embedding=EMBEDDING] [--config=FILE]
    memory-engine migrate --from=SOURCE --to=TARGET [--verify] [--cleanup]
    memory-engine export --format=FORMAT --output=FILE [--include-embeddings] [--include-metadata]
    memory-engine import --file=FILE [--format=FORMAT] [--merge-duplicates] [--update-existing]
    memory-engine backup [--strategy=STRATEGY] [--compression=TYPE] [--output=FILE]
    memory-engine restore --backup=ID [--clear-existing]
    memory-engine health-check [--detailed] [--format=FORMAT]
    memory-engine status [--format=FORMAT]
    memory-engine plugins list [--type=TYPE]
    memory-engine plugins install NAME [--version=VERSION]
    memory-engine plugins uninstall NAME
    memory-engine config show [--section=SECTION]
    memory-engine config set KEY VALUE
    memory-engine config validate
    memory-engine version
    memory-engine --help

Commands:
    init                Initialize new Memory Engine instance
    migrate             Migrate data between backends
    export              Export knowledge graph data
    import              Import knowledge graph data
    backup              Create backup of knowledge graph
    restore             Restore from backup
    health-check        Check system health
    status              Show system status
    plugins             Manage plugins
    config              Manage configuration
    version             Show version information

Options:
    -h --help           Show this help message
    --backend=BACKEND   Storage backend [default: sqlite]
    --embedding=EMBED   Embedding provider [default: sentence_transformers]
    --config=FILE       Configuration file path
    --from=SOURCE       Source backend for migration
    --to=TARGET         Target backend for migration
    --verify            Verify migration integrity
    --cleanup           Clean up source after migration
    --format=FORMAT     Output format (json, csv, xml, graphml, etc.)
    --output=FILE       Output file path
    --include-embeddings Include embeddings in export
    --include-metadata  Include metadata in export
    --file=FILE         Input file path
    --merge-duplicates  Merge duplicate entries during import
    --update-existing   Update existing entries during import
    --strategy=STRATEGY Backup strategy (full, incremental, differential)
    --compression=TYPE  Compression type (none, gzip, bzip2, lzma)
    --backup=ID         Backup ID for restore
    --clear-existing    Clear existing data before restore
    --detailed          Show detailed information
    --type=TYPE         Plugin type filter
    --version=VERSION   Plugin version
    --section=SECTION   Configuration section
"""

import os
import sys
import asyncio
import json
import yaml
from typing import Dict, List, Any, Optional
import logging
import traceback
from datetime import datetime

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from memory_core.config.config_manager import ConfigManager
    from memory_core.core.knowledge_engine import KnowledgeEngine
    from memory_core.migration.backend_migrator import BackendMigrator, MigrationConfig, MigrationStrategy
    from memory_core.migration.data_exporter import DataExporter, ExportFormat, ExportConfig
    from memory_core.migration.data_importer import DataImporter, ImportConfig
    from memory_core.migration.backup_manager import BackupManager, BackupConfig, BackupStrategy, CompressionType
    from memory_core.health.health_checker import HealthChecker
    from memory_core.health.service_monitor import ServiceMonitor
    from memory_core.health.health_endpoints import HealthEndpoints
    from memory_core.plugins.plugin_manager import PluginManager
    from memory_core.plugins.plugin_registry import PluginRegistry
    from memory_core.performance.metrics_collector import MetricsCollector, PerformanceMonitor
except ImportError as e:
    print(f"Error importing Memory Engine modules: {e}")
    print("Make sure you're running from the correct directory and all dependencies are installed.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryEngineCLI:
    """Memory Engine command line interface."""
    
    def __init__(self):
        self.config_manager = None
        self.engine = None
        self.plugin_manager = None
        self.health_checker = None
        self.service_monitor = None
    
    async def initialize(self, config_file: str = None):
        """Initialize CLI components."""
        try:
            # Load configuration
            self.config_manager = ConfigManager(config_file)
            
            # Initialize plugin manager
            self.plugin_manager = PluginManager()
            await self.plugin_manager.discover_plugins()
            
            # Initialize health monitoring
            self.health_checker = HealthChecker()
            self.service_monitor = ServiceMonitor()
            
            logger.info("CLI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLI: {e}")
            raise
    
    async def init_command(self, backend: str = "sqlite", embedding: str = "sentence_transformers", 
                          config_file: str = None):
        """Initialize new Memory Engine instance."""
        print("🚀 Initializing Memory Engine...")
        
        try:
            # Create default configuration
            config = {
                'storage': {
                    'backend': backend,
                    backend: {}
                },
                'embeddings': {
                    'provider': embedding,
                    embedding: {}
                },
                'llm': {
                    'provider': 'gemini',  # Default
                    'gemini': {}
                }
            }
            
            # Write configuration file
            config_path = config_file or 'config/config.yaml'
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, indent=2)
            
            print(f"✅ Configuration created: {config_path}")
            print(f"📊 Storage backend: {backend}")
            print(f"🔤 Embedding provider: {embedding}")
            print("")
            print("Next steps:")
            print("1. Update configuration with API keys and settings")
            print("2. Run 'memory-engine health-check' to verify setup")
            print("3. Import your data or start using the knowledge engine")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            sys.exit(1)
    
    async def migrate_command(self, from_backend: str, to_backend: str, 
                             verify: bool = False, cleanup: bool = False):
        """Migrate data between backends."""
        print(f"🔄 Migrating from {from_backend} to {to_backend}...")
        
        try:
            await self.initialize()
            
            # Create source and target engines
            source_config = self.config_manager.config.copy()
            source_config['storage']['backend'] = from_backend
            
            target_config = self.config_manager.config.copy()
            target_config['storage']['backend'] = to_backend
            
            source_engine = KnowledgeEngine(config=source_config)
            target_engine = KnowledgeEngine(config=target_config)
            
            await source_engine.initialize()
            await target_engine.initialize()
            
            # Configure migration
            migration_config = MigrationConfig(
                source_backend=from_backend,
                target_backend=to_backend,
                strategy=MigrationStrategy.INCREMENTAL,
                verify_migration=verify,
                cleanup_source=cleanup,
                progress_callback=self._migration_progress
            )
            
            # Perform migration
            migrator = BackendMigrator(migration_config)
            result = await migrator.migrate(source_engine, target_engine)
            
            if result.success:
                print(f"✅ Migration completed successfully")
                print(f"📊 Migrated {result.target_count} items in {result.duration:.2f}s")
            else:
                print(f"❌ Migration failed")
                for error in result.errors:
                    print(f"   Error: {error}")
                sys.exit(1)
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            sys.exit(1)
    
    async def export_command(self, format: str, output: str, 
                           include_embeddings: bool = False, 
                           include_metadata: bool = True):
        """Export knowledge graph data."""
        print(f"📤 Exporting data in {format} format to {output}...")
        
        try:
            await self.initialize()
            
            # Initialize engine
            self.engine = KnowledgeEngine()
            await self.engine.initialize()
            
            # Configure export
            export_config = ExportConfig(
                format=ExportFormat(format),
                include_embeddings=include_embeddings,
                include_metadata=include_metadata
            )
            
            # Perform export
            exporter = DataExporter(export_config)
            result = await exporter.export_knowledge_graph(self.engine, output)
            
            if result['success']:
                print(f"✅ Export completed successfully")
                print(f"📊 Exported {result['nodes_exported']} nodes and {result['relationships_exported']} relationships")
                print(f"📁 Output: {result['output_path']}")
            else:
                print(f"❌ Export failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            sys.exit(1)
    
    async def import_command(self, file: str, format: str = None, 
                           merge_duplicates: bool = False, 
                           update_existing: bool = False):
        """Import knowledge graph data."""
        print(f"📥 Importing data from {file}...")
        
        try:
            await self.initialize()
            
            # Initialize engine
            self.engine = KnowledgeEngine()
            await self.engine.initialize()
            
            # Configure import
            import_config = ImportConfig(
                merge_duplicates=merge_duplicates,
                update_existing=update_existing
            )
            
            # Perform import
            importer = DataImporter(import_config)
            result = await importer.import_from_file(file, self.engine, format)
            
            if result['success']:
                print(f"✅ Import completed successfully")
                print(f"📊 Imported {result['imported_count']} items")
                if result['error_count'] > 0:
                    print(f"⚠️  {result['error_count']} errors occurred")
            else:
                print(f"❌ Import failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            sys.exit(1)
    
    async def backup_command(self, strategy: str = "full", 
                           compression: str = "gzip", output: str = None):
        """Create backup of knowledge graph."""
        print(f"💾 Creating {strategy} backup...")
        
        try:
            await self.initialize()
            
            # Initialize engine
            self.engine = KnowledgeEngine()
            await self.engine.initialize()
            
            # Configure backup
            backup_config = BackupConfig(
                strategy=BackupStrategy(strategy),
                compression=CompressionType(compression),
                progress_callback=self._backup_progress
            )
            
            # Create backup directory
            backup_dir = output or "backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Perform backup
            backup_manager = BackupManager(backup_config, backup_dir)
            backup_info = await backup_manager.create_backup(self.engine)
            
            print(f"✅ Backup completed successfully")
            print(f"📊 Backup ID: {backup_info.backup_id}")
            print(f"📁 File: {backup_info.file_path}")
            print(f"📦 Size: {backup_info.file_size_mb:.2f} MB")
            print(f"🔢 Nodes: {backup_info.node_count}")
            print(f"🔗 Relationships: {backup_info.relationship_count}")
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            sys.exit(1)
    
    async def restore_command(self, backup_id: str, clear_existing: bool = False):
        """Restore from backup."""
        print(f"♻️  Restoring from backup {backup_id}...")
        
        try:
            await self.initialize()
            
            # Initialize engine
            self.engine = KnowledgeEngine()
            await self.engine.initialize()
            
            # Configure backup manager
            backup_config = BackupConfig(progress_callback=self._restore_progress)
            backup_manager = BackupManager(backup_config, "backups")
            
            # Perform restore
            result = await backup_manager.restore_backup(backup_id, self.engine, clear_existing)
            
            if result['success']:
                print(f"✅ Restore completed successfully")
                print(f"📊 Restored {result['restored_nodes']} nodes and {result['restored_relationships']} relationships")
            else:
                print(f"❌ Restore failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
            
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            sys.exit(1)
    
    async def health_check_command(self, detailed: bool = False, format: str = "text"):
        """Check system health."""
        print("🏥 Checking system health...")
        
        try:
            await self.initialize()
            
            # Run health checks
            config = self.config_manager.config
            health_endpoints = HealthEndpoints(
                self.health_checker, 
                self.service_monitor
            )
            
            if detailed:
                result = await health_endpoints.detailed_status(config)
            else:
                result = await health_endpoints.health_check(include_details=True)
            
            # Format output
            if format == "json":
                print(json.dumps(result, indent=2, default=str))
            else:
                self._print_health_status(result, detailed)
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            sys.exit(1)
    
    async def status_command(self, format: str = "text"):
        """Show system status."""
        print("📊 System Status")
        print("=" * 50)
        
        try:
            await self.initialize()
            
            # Get configuration info
            config = self.config_manager.config
            
            print(f"Storage Backend: {config.get('storage', {}).get('backend', 'unknown')}")
            print(f"Embedding Provider: {config.get('embeddings', {}).get('provider', 'unknown')}")
            print(f"LLM Provider: {config.get('llm', {}).get('provider', 'unknown')}")
            
            # Get health status
            health_result = await self.health_checker.run_all_checks() if self.health_checker else {}
            overall_status = self.health_checker.get_overall_status() if self.health_checker else "unknown"
            
            print(f"Overall Health: {overall_status.value if hasattr(overall_status, 'value') else overall_status}")
            print(f"Active Checks: {len(health_result)}")
            
            # Get plugin info
            if self.plugin_manager:
                available_plugins = len(self.plugin_manager.list_available_plugins())
                loaded_plugins = len(self.plugin_manager.list_loaded_plugins())
                print(f"Plugins: {loaded_plugins}/{available_plugins} loaded")
            
            print("")
            print("For detailed health information, run: memory-engine health-check --detailed")
            
        except Exception as e:
            print(f"❌ Status check failed: {e}")
            sys.exit(1)
    
    async def plugins_command(self, action: str, name: str = None, 
                            plugin_type: str = None, version: str = None):
        """Manage plugins."""
        if action == "list":
            await self._list_plugins(plugin_type)
        elif action == "install":
            await self._install_plugin(name, version)
        elif action == "uninstall":
            await self._uninstall_plugin(name)
        else:
            print(f"❌ Unknown plugin action: {action}")
            sys.exit(1)
    
    async def _list_plugins(self, plugin_type: str = None):
        """List available plugins."""
        print("🔌 Available Plugins")
        print("=" * 50)
        
        try:
            await self.initialize()
            
            available_plugins = self.plugin_manager.list_available_plugins()
            loaded_plugins = self.plugin_manager.list_loaded_plugins()
            
            if plugin_type:
                available_plugins = [p for p in available_plugins if p.plugin_type == plugin_type]
            
            for plugin in available_plugins:
                status = "✅ Loaded" if plugin.is_loaded else "⭕ Available"
                print(f"{status} {plugin.name} v{plugin.version} ({plugin.plugin_type})")
                print(f"    {plugin.description}")
                print(f"    Author: {plugin.author}")
                print("")
            
            if not available_plugins:
                print("No plugins found")
            
        except Exception as e:
            print(f"❌ Failed to list plugins: {e}")
            sys.exit(1)
    
    async def _install_plugin(self, name: str, version: str = None):
        """Install a plugin."""
        print(f"📦 Installing plugin {name}...")
        
        try:
            await self.initialize()
            
            # Load plugin
            plugin = await self.plugin_manager.load_plugin(name)
            
            if plugin:
                print(f"✅ Plugin {name} installed successfully")
            else:
                print(f"❌ Failed to install plugin {name}")
                sys.exit(1)
            
        except Exception as e:
            print(f"❌ Plugin installation failed: {e}")
            sys.exit(1)
    
    async def _uninstall_plugin(self, name: str):
        """Uninstall a plugin."""
        print(f"🗑️  Uninstalling plugin {name}...")
        
        try:
            await self.initialize()
            
            # Unload plugin
            success = await self.plugin_manager.unload_plugin(name)
            
            if success:
                print(f"✅ Plugin {name} uninstalled successfully")
            else:
                print(f"❌ Failed to uninstall plugin {name}")
                sys.exit(1)
            
        except Exception as e:
            print(f"❌ Plugin uninstallation failed: {e}")
            sys.exit(1)
    
    async def config_command(self, action: str, key: str = None, 
                           value: str = None, section: str = None):
        """Manage configuration."""
        if action == "show":
            await self._show_config(section)
        elif action == "set":
            await self._set_config(key, value)
        elif action == "validate":
            await self._validate_config()
        else:
            print(f"❌ Unknown config action: {action}")
            sys.exit(1)
    
    async def _show_config(self, section: str = None):
        """Show configuration."""
        try:
            await self.initialize()
            
            config = self.config_manager.config
            
            if section:
                config = config.get(section, {})
                print(f"📋 Configuration - {section}")
            else:
                print("📋 Configuration")
            
            print("=" * 50)
            print(yaml.dump(config, indent=2))
            
        except Exception as e:
            print(f"❌ Failed to show config: {e}")
            sys.exit(1)
    
    async def _set_config(self, key: str, value: str):
        """Set configuration value."""
        print(f"⚙️  Setting {key} = {value}")
        
        try:
            await self.initialize()
            
            # Parse key path (e.g., "storage.backend")
            keys = key.split('.')
            config = self.config_manager.config
            
            # Navigate to parent
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set value
            config[keys[-1]] = value
            
            # Save configuration
            await self.config_manager.save_config()
            
            print(f"✅ Configuration updated")
            
        except Exception as e:
            print(f"❌ Failed to set config: {e}")
            sys.exit(1)
    
    async def _validate_config(self):
        """Validate configuration."""
        print("✅ Validating configuration...")
        
        try:
            await self.initialize()
            
            # Basic validation
            config = self.config_manager.config
            errors = []
            
            # Check required sections
            required_sections = ['storage', 'embeddings', 'llm']
            for section in required_sections:
                if section not in config:
                    errors.append(f"Missing required section: {section}")
            
            # Validate storage backend
            storage = config.get('storage', {})
            backend = storage.get('backend')
            if backend and backend not in storage:
                errors.append(f"Storage backend '{backend}' not configured")
            
            if errors:
                print("❌ Configuration validation failed:")
                for error in errors:
                    print(f"   {error}")
                sys.exit(1)
            else:
                print("✅ Configuration is valid")
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            sys.exit(1)
    
    def version_command(self):
        """Show version information."""
        print("Memory Engine CLI v0.4.0")
        print("Production Readiness Release")
        print("")
        print("Features:")
        print("• Advanced caching and performance optimization")
        print("• Comprehensive health monitoring")
        print("• Backend migration and data export/import")
        print("• Plugin architecture for custom extensions")
        print("• CLI management tools")
    
    def _migration_progress(self, progress: float, status: str):
        """Migration progress callback."""
        print(f"🔄 {progress:.1f}% - {status}")
    
    def _backup_progress(self, progress: float, status: str):
        """Backup progress callback."""
        print(f"💾 {progress:.1f}% - {status}")
    
    def _restore_progress(self, progress: float, status: str):
        """Restore progress callback."""
        print(f"♻️  {progress:.1f}% - {status}")
    
    def _print_health_status(self, result: Dict[str, Any], detailed: bool = False):
        """Print health status in human-readable format."""
        overall_status = result.get('overall_status', 'unknown')
        
        # Status emoji
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'unhealthy': '❌',
            'unknown': '❓'
        }
        
        print(f"{status_emoji.get(overall_status, '❓')} Overall Status: {overall_status.upper()}")
        print("")
        
        if 'checks' in result:
            print("Component Health:")
            print("-" * 30)
            
            for name, check_result in result['checks'].items():
                status = check_result.get('status', 'unknown')
                message = check_result.get('message', '')
                response_time = check_result.get('response_time', 0)
                
                emoji = status_emoji.get(status, '❓')
                print(f"{emoji} {name}: {status}")
                
                if detailed:
                    print(f"    Message: {message}")
                    print(f"    Response Time: {response_time:.3f}s")
                    
                    details = check_result.get('details')
                    if details:
                        print(f"    Details: {json.dumps(details, indent=6)}")
                    print("")
        
        if 'dependencies' in result:
            deps = result['dependencies']
            print(f"Dependencies: {deps.get('healthy_services', 0)}/{deps.get('total_services', 0)} healthy")


def parse_args():
    """Parse command line arguments manually."""
    import sys
    
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    args = {}
    
    # Parse remaining arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg.startswith('--'):
            if '=' in arg:
                key, value = arg[2:].split('=', 1)
                args[key] = value
            else:
                key = arg[2:]
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                    args[key] = sys.argv[i + 1]
                    i += 1
                else:
                    args[key] = True
        else:
            # Positional argument
            if 'positional' not in args:
                args['positional'] = []
            args['positional'].append(arg)
        
        i += 1
    
    return command, args


async def main():
    """Main CLI entry point."""
    try:
        command, args = parse_args()
        cli = MemoryEngineCLI()
        
        if command == "init":
            await cli.init_command(
                backend=args.get('backend', 'sqlite'),
                embedding=args.get('embedding', 'sentence_transformers'),
                config_file=args.get('config')
            )
        
        elif command == "migrate":
            if 'from' not in args or 'to' not in args:
                print("❌ Migration requires --from and --to arguments")
                sys.exit(1)
            
            await cli.migrate_command(
                from_backend=args['from'],
                to_backend=args['to'],
                verify=args.get('verify', False),
                cleanup=args.get('cleanup', False)
            )
        
        elif command == "export":
            if 'format' not in args or 'output' not in args:
                print("❌ Export requires --format and --output arguments")
                sys.exit(1)
            
            await cli.export_command(
                format=args['format'],
                output=args['output'],
                include_embeddings=args.get('include-embeddings', False),
                include_metadata=args.get('include-metadata', True)
            )
        
        elif command == "import":
            if 'file' not in args:
                print("❌ Import requires --file argument")
                sys.exit(1)
            
            await cli.import_command(
                file=args['file'],
                format=args.get('format'),
                merge_duplicates=args.get('merge-duplicates', False),
                update_existing=args.get('update-existing', False)
            )
        
        elif command == "backup":
            await cli.backup_command(
                strategy=args.get('strategy', 'full'),
                compression=args.get('compression', 'gzip'),
                output=args.get('output')
            )
        
        elif command == "restore":
            if 'backup' not in args:
                print("❌ Restore requires --backup argument")
                sys.exit(1)
            
            await cli.restore_command(
                backup_id=args['backup'],
                clear_existing=args.get('clear-existing', False)
            )
        
        elif command == "health-check":
            await cli.health_check_command(
                detailed=args.get('detailed', False),
                format=args.get('format', 'text')
            )
        
        elif command == "status":
            await cli.status_command(
                format=args.get('format', 'text')
            )
        
        elif command == "plugins":
            if 'positional' not in args or len(args['positional']) == 0:
                print("❌ Plugins command requires action (list, install, uninstall)")
                sys.exit(1)
            
            action = args['positional'][0]
            name = args['positional'][1] if len(args['positional']) > 1 else None
            
            await cli.plugins_command(
                action=action,
                name=name,
                plugin_type=args.get('type'),
                version=args.get('version')
            )
        
        elif command == "config":
            if 'positional' not in args or len(args['positional']) == 0:
                print("❌ Config command requires action (show, set, validate)")
                sys.exit(1)
            
            action = args['positional'][0]
            key = args['positional'][1] if len(args['positional']) > 1 else None
            value = args['positional'][2] if len(args['positional']) > 2 else None
            
            await cli.config_command(
                action=action,
                key=key,
                value=value,
                section=args.get('section')
            )
        
        elif command == "version":
            cli.version_command()
        
        elif command in ["--help", "-h", "help"]:
            print(__doc__)
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Run 'memory-engine --help' for usage information")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if os.getenv('DEBUG'):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())