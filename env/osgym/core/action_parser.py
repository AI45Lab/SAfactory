"""
Action Parser Module

Parses agent response strings into executable action commands.
Handles special commands (WAIT, DONE, FAIL) and code block extraction.
"""

import re
from typing import List, Tuple, Optional


class ActionParser:
    """
    Parses agent response strings into executable action commands.

    Supports:
    - Special commands: WAIT, DONE, FAIL
    - Code blocks with optional language identifiers
    - Multiple actions separated by semicolons
    """

    SPECIAL_COMMANDS = {"WAIT", "DONE", "FAIL"}

    @staticmethod
    def parse_actions(action_str: str) -> List[str]:
        """
        Parse action string into list of executable commands.

        Args:
            action_str: Raw action string from agent response

        Returns:
            List of parsed action commands
        """
        if not action_str:
            return []

        normalized = "\n".join([line.strip() for line in action_str.split(';') if line.strip()])
        special_cmds = ActionParser.SPECIAL_COMMANDS

        if normalized in special_cmds:
            return [normalized]

        # Check for ```DONE```, ```FAIL```, ```WAIT``` format special commands
        special_pattern = r"```\s*(DONE|FAIL|WAIT)\s*```"
        special_matches = re.findall(special_pattern, action_str, re.IGNORECASE)
        if special_matches:
            return [m.upper() for m in special_matches]

        # Regex to match code blocks, requiring whitespace/newline after language identifier
        pattern = r"```(?:(\w+)\s+)?(.*?)```"
        matches = re.findall(pattern, action_str, re.DOTALL)
        commands: List[str] = []

        if matches:
            for lang, match in matches:
                snippet = match.strip()
                if not snippet:
                    continue
                last_line = snippet.splitlines()[-1].strip() if snippet.splitlines() else ""
                if snippet in special_cmds:
                    commands.append(snippet)
                    continue
                if last_line in special_cmds:
                    body = "\n".join(snippet.splitlines()[:-1]).strip()
                    if body:
                        commands.append(body)
                    commands.append(last_line)
                else:
                    commands.append(snippet)
        else:
            upper_text = action_str.upper()
            if re.search(r'\bDONE\b', upper_text):
                commands.append("DONE")
            elif re.search(r'\bFAIL\b', upper_text):
                commands.append("FAIL")
            elif re.search(r'\bWAIT\b', upper_text):
                commands.append("WAIT")
            elif normalized:
                commands.append(normalized)

        if not commands:
            stripped = action_str.strip()
            triple_match = re.fullmatch(r"`{3}(?:\w+)?\s*(DONE|FAIL|WAIT)\s*`{3}", stripped, re.IGNORECASE)
            if triple_match:
                commands.append(triple_match.group(1).upper())

        return [cmd for cmd in commands if cmd]

    @staticmethod
    def strip_special_command(actions: List[str]) -> Tuple[List[str], Optional[str]]:
        """
        Separate special commands from regular actions.

        Args:
            actions: List of action commands

        Returns:
            Tuple of (cleaned_actions, special_command)
            where special_command is DONE/FAIL/WAIT if present, None otherwise
        """
        special_cmd: Optional[str] = None
        cleaned: List[str] = []

        for cmd in actions:
            normalized = cmd.strip().upper()
            if normalized in {"DONE", "FAIL"}:
                special_cmd = normalized
                continue
            if normalized == "WAIT" and len(actions) == 1:
                special_cmd = normalized
                continue
            cleaned.append(cmd)

        return cleaned, special_cmd

    @staticmethod
    def is_special_command(action: str) -> bool:
        """Check if action is a special command."""
        return action.strip().upper() in ActionParser.SPECIAL_COMMANDS
