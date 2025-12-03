# PiRadar Documentation

This directory contains the Quarto-based documentation for PiRadar.

## Building the Documentation

### Prerequisites

Install Quarto from https://quarto.org/docs/get-started/

### Preview Documentation

```bash
quarto preview docs/
```

This will start a local server at http://localhost:4200

### Render Documentation

```bash
quarto render docs/
```

Output will be in `docs/_book/`

## Documentation Structure

The documentation is organized as a Quarto book with the following structure:

### Part 1: Getting Started
- **index.qmd**: Introduction and overview
- **installation.qmd**: Installation instructions
- **quickstart.qmd**: Quick start guide

### Part 2: Hardware
- **hardware-overview.qmd**: Radar hardware overview
- **hardware-specs.qmd**: Detailed hardware specifications

### Part 3: Examples
- **examples.qmd**: Basic usage examples
- **examples-zmq.qmd**: ZeroMQ examples
- **examples-adaptive.qmd**: Adaptive radar examples

### Part 4: Network & Data
- **network-format.qmd**: Network data format specification
- **zmq-communication.qmd**: ZeroMQ communication details

### Part 5: Programming the Radar
- **configuration.qmd**: Configuration file format
- **register-map.qmd**: Register map overview
- **direct-control.qmd**: Direct register control

### Part 6: API Reference
- **cli-reference.qmd**: Command-line interface reference
- **python-api.qmd**: Python API reference

### Part 7: Advanced Topics
- **adaptive-radar.qmd**: Adaptive radar operation

### Part 8: Development
- **development.qmd**: Development guide

## Configuration

The main configuration file is `_quarto.yml`, which defines:

- Book structure and chapters
- Output formats (HTML, PDF)
- Theme and styling
- Navigation

## Custom Styling

Custom CSS is in `styles.css` and provides:

- Dark/light mode support
- Code block styling
- Table formatting
- Custom callout boxes

## References

Bibliography entries are in `references.bib` for citing external resources.

## Legacy Files

The following markdown files are from the old MkDocs setup and can be removed:

- `adaptive-radar.md` → Converted to `adaptive-radar.qmd`
- `cli.md` → Converted to `cli-reference.qmd`
- `configuration.md` → Converted to `configuration.qmd`
- `development.md` → Converted to `development.qmd`
- `getting-started.md` → Split into `installation.qmd` and `quickstart.qmd`
- `index.md` → Converted to `index.qmd`
- `registermap.md` → Converted to `register-map.qmd`
- `zmq.md` → Converted to `zmq-communication.qmd` and `network-format.qmd`

The old `mkdocs.yml` file can also be removed.

## Publishing

To publish the documentation:

1. Render the documentation: `quarto render docs/`
2. The output in `docs/_book/` can be deployed to:
   - GitHub Pages
   - Netlify
   - Any static hosting service

### GitHub Pages

Add to your repository's `.github/workflows/quarto-publish.yml`:

```yaml
name: Publish Quarto Documentation

on:
  push:
    branches: [main]

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Quarto
        uses: quarto-dev/quarto-actions/setup@v2
      
      - name: Render Quarto Documentation
        run: quarto render docs/
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_book
```

## Contributing

When adding new documentation:

1. Create a new `.qmd` file in the appropriate section
2. Add the file to `_quarto.yml` in the correct chapter
3. Use Quarto markdown syntax (supports LaTeX math, callouts, etc.)
4. Preview changes with `quarto preview docs/`
5. Submit a pull request

## Support

For issues with the documentation:

- Check https://quarto.org/docs/ for Quarto-specific questions
- Open an issue at https://github.com/juhasch/piradar/issues

