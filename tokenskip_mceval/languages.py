"""McEval language → markdown fence tag + human-friendly name.

Keys are McEval's language identifiers (matching the *.jsonl filenames in
McEval/data/). Values are the canonical markdown code-fence tag we'll use in
prompts and look for in model outputs.
"""

LANG_TO_FENCE = {
    "AWK": "awk",
    "C": "c",
    "CPP": "cpp",
    "C#": "csharp",
    "CoffeeScript": "coffeescript",
    "Common Lisp": "lisp",
    "Dart": "dart",
    "Elixir": "elixir",
    "Emacs Lisp": "elisp",
    "Erlang": "erlang",
    "F#": "fsharp",
    "Fortran": "fortran",
    "Go": "go",
    "Groovy": "groovy",
    "Haskell": "haskell",
    "HTML": "html",
    "Java": "java",
    "JavaScript": "javascript",
    "JSON": "json",
    "Julia": "julia",
    "Kotlin": "kotlin",
    "Lua": "lua",
    "Markdown": "markdown",
    "Pascal": "pascal",
    "Perl": "perl",
    "PHP": "php",
    "PowerShell": "powershell",
    "Python": "python",
    "R": "r",
    "Racket": "racket",
    "Ruby": "ruby",
    "Rust": "rust",
    "Scala": "scala",
    "Scheme": "scheme",
    "Shell": "bash",
    "Swift": "swift",
    "Tcl": "tcl",
    "TypeScript": "typescript",
    "VimScript": "vim",
    "Visual Basic": "vbnet",
}


def jsonl_basename(lang: str) -> str:
    """Map a language key to the *.jsonl filename used in McEval/data/."""
    return f"{lang}.jsonl"
