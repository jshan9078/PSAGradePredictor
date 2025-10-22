# Documentation Index

Comprehensive documentation for the PSA Card Grading Estimator project.

## Quick Links

- [← Back to Main README](../README.md)
- [Setup Guide](../SETUP.md) - Get started quickly
- [Security Guide](../SECURITY.md) - Credential and secret management
- [Training History](../CHANGELOG.md) - Complete model evolution log

## Documentation Files

### Getting Started

1. **[quickstart.md](quickstart.md)** - Fast-track guide to deploy and train
   - Minimal steps to get running
   - Quick command reference
   - Common workflows

### Deployment & Infrastructure

2. **[deployment.md](deployment.md)** - Comprehensive deployment guide
   - Vertex AI setup and configuration
   - GCS bucket management
   - Docker image building and pushing
   - Training job submission
   - Monitoring and debugging

3. **[development.md](development.md)** - Local development workflow
   - Setting up local environment
   - Running tests locally
   - Debugging training code
   - Testing without GCS access

### Architecture & Design

4. **[architecture.md](architecture.md)** - Model architecture deep dive
   - Dual-branch ResNet design
   - Why ResNet-18 front + ResNet-34 back
   - Loss function design (CE + EMD)
   - Feature fusion strategies
   - Architecture alternatives considered

5. **[technical-paper.tex](technical-paper.tex)** - Academic-style technical paper
   - LaTeX document with mathematical formulations
   - Theoretical justification for design choices
   - Complete problem formulation
   - Loss derivations and proofs

### Technical Details

6. **[image-resizing.md](image-resizing.md)** - Image preprocessing explained
   - LAB color space conversion
   - CLAHE enhancement
   - Edge detection (Sobel, Laplacian)
   - Why these preprocessing steps matter

7. **[image-size-guide.md](image-size-guide.md)** - Image size optimization
   - Why 384×384 was chosen
   - Trade-offs between size and performance
   - Memory considerations
   - Batch size implications

## Document Organization

### Root Level Docs

These stay in the project root for easy access:

- **README.md** - Main project overview (you are here!)
- **SETUP.md** - Environment setup and configuration
- **SECURITY.md** - Security best practices and credential management
- **CHANGELOG.md** - Complete training history with metrics and lessons

### Detailed Docs (docs/ folder)

In-depth technical documentation for specific aspects of the project.

## Reading Path by Role

### For New Contributors

1. [Main README](../README.md) → Overview and quick start
2. [SETUP.md](../SETUP.md) → Set up your environment
3. [SECURITY.md](../SECURITY.md) → Learn security practices
4. [quickstart.md](quickstart.md) → Run your first training job
5. [development.md](development.md) → Set up local development

### For ML Engineers

1. [Main README](../README.md) → Project overview
2. [CHANGELOG.md](../CHANGELOG.md) → What's been tried
3. [architecture.md](architecture.md) → Model design
4. [deployment.md](deployment.md) → Training infrastructure
5. [image-resizing.md](image-resizing.md) → Preprocessing details

### For DevOps/Infrastructure

1. [SETUP.md](../SETUP.md) → Environment configuration
2. [SECURITY.md](../SECURITY.md) → Secrets management
3. [deployment.md](deployment.md) → GCP infrastructure
4. [development.md](development.md) → Testing and debugging

### For Researchers

1. [CHANGELOG.md](../CHANGELOG.md) → Experimental history
2. [technical-paper.tex](technical-paper.tex) → Mathematical formulation
3. [architecture.md](architecture.md) → Design rationale
4. [image-resizing.md](image-resizing.md) → Preprocessing theory

## Contributing to Documentation

When adding or modifying documentation:

1. **Keep README.md concise** - Move detailed content to docs/
2. **Update this index** when adding new docs
3. **Link between related docs** for easy navigation
4. **Use consistent formatting** (markdown, code blocks, etc.)
5. **Add to CHANGELOG.md** for training/model changes
6. **Include code examples** where applicable

### Documentation Standards

- Use **clear headings** with proper hierarchy
- Include **code examples** with syntax highlighting
- Add **cross-references** to related sections
- Keep **command outputs** in code blocks
- Use **tables** for structured data
- Include **warnings/notes** for important information

## Need Help?

- Check [SETUP.md](../SETUP.md) for environment issues
- See [deployment.md](deployment.md) for GCP problems
- Review [CHANGELOG.md](../CHANGELOG.md) for training insights
- Read [SECURITY.md](../SECURITY.md) for credential questions

## External Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Google Cloud Storage](https://cloud.google.com/storage/docs)
- [Docker Documentation](https://docs.docker.com/)
