# Claude Code Configuration Template

This directory contains example Claude Code configuration files for the Memory Engine project.

## Setup for Claude Code Users

1. Copy this directory to `.claude` in your project root:
   ```bash
   cp -r .claude.example .claude
   ```

2. Customize the files in `.claude/` for your specific setup:
   - Edit `CLAUDE.md` with your environment details
   - Modify `settings.json` for your preferences
   - Update command scripts as needed

3. The `.claude` directory is gitignored to keep your personal configuration private.

## Files Included

- `CLAUDE.md` - Comprehensive project guide for Claude Code
- `settings.json` - Tool preferences and workflow configuration
- `commands/` - Custom slash commands for development workflows
- `examples/` - Template configurations

## Custom Commands

- `/setup` - Complete development environment setup
- `/test` - Comprehensive test suite with service checks
- `/docker` - Docker service management
- `/commit` - Smart commit with testing and formatting

Feel free to modify these files to match your development style and environment!