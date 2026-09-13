# Paper source

`main.tex` is the only entry point. The directory is organised by
responsibility:

```text
latex/
├── main.tex             document setup and section order
├── commands.tex         paper-specific notation only
├── references.bib       paper bibliography
├── sections/            paper content
├── tables/              standalone table sources
├── style/               bundled ICLR template dependencies
└── build/               generated files; ignored by Git
```

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The PDF is written to `build/main.pdf`; auxiliary files stay in `build/` as
well. Remove them with:

```bash
latexmk -C main.tex
```

Bibliography commands in `main.tex` are commented out while the paper has no
citations. Re-enable them after adding the first real entry to `references.bib`.
