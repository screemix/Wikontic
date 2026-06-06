"""
Fix for trl's encoding issue on Windows.
Pre-loads trl's template files with UTF-8 encoding to prevent cp1252 decode errors.
"""

from pathlib import Path
import sys


def fix_trl_encoding():
    """
    Pre-read trl's chat template files with UTF-8 encoding.
    This prevents trl from trying to read them later with the wrong encoding.
    """
    try:
        # Find trl installation path WITHOUT importing trl
        # (importing trl would trigger the encoding error we're trying to fix)
        import importlib.util
        
        spec = importlib.util.find_spec("trl")
        if spec is None or spec.origin is None:
            return
            
        trl_path = Path(spec.origin).parent
        # Correct path: trl/chat_templates/ not trl/data/chat_templates/
        templates_dir = trl_path / "chat_templates"

        if not templates_dir.exists():
            return  # trl structure might be different

        # Pre-read all jinja files with UTF-8
        template_cache = {}
        for jinja_file in templates_dir.glob("*.jinja"):
            try:
                template_cache[jinja_file.name] = jinja_file.read_text(encoding="utf-8")
            except Exception:
                pass

        if not template_cache:
            return

        # Now monkey-patch Path.read_text ONLY for trl's template directory
        original_read_text = Path.read_text

        def patched_read_text(self, encoding=None, errors=None):
            # Check if this is a trl template file
            try:
                resolved = self.resolve()
                if resolved.parent == templates_dir.resolve():
                    filename = resolved.name
                    if filename in template_cache:
                        return template_cache[filename]
            except Exception:
                pass
            # Otherwise, use original with explicit UTF-8 default
            if encoding is None:
                encoding = "utf-8"
            return original_read_text(self, encoding=encoding, errors=errors)

        Path.read_text = patched_read_text

    except Exception:
        # If anything fails, don't break the import
        pass


# Apply fix immediately when module is imported
fix_trl_encoding()
