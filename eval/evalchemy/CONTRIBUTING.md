# Contributing to Evalchemy

Welcome to the Evalchemy project! We value your contributions and are excited to have you collaborate with us. This document outlines the guidelines for contributing to ensure a smooth and productive process.

## TLDR;

Get setup with
```
conda create -n evalchemy python=3.10
conda activate evalchemy
make install
```

Refer to the [instructions on how to add an evaluation benchmark](https://github.com/mlfoundations/evalchemy?tab=readme-ov-file#%EF%B8%8F-implementing-custom-evaluations)

Add tested eval metrics against publicly reported numbers in [reproduced_benchmarks.md](reproduced_benchmarks.md).

Add eval name to the list in [README.md](README.md)

## Table of Contents

1. [Setting Up the Project](#setting-up-the-project)
2. [How to Contribute](#how-to-contribute)
3. [Submitting Changes](#submitting-changes)
4. [Issue Reporting](#issue-reporting)
5. [Pull Request Guidelines](#pull-request-guidelines)

---

## Setting Up the Project

```bash
conda create -n evalchemy python=3.10
conda activate evalchemy
make install
```

This will create a virtual environment and install the dependencies, and pre-commit hooks.

## How to Contribute

1. **Fork the Repository**: Start by forking the repository to your GitHub account.
2. **Clone Your Fork**: Clone your fork to your local machine using:
   ```bash
   git clone https://github.com/<username>/evalchemy.git
   ```
3. **Create an Issue**: Before starting any work, create an issue in the repository to discuss your proposed changes or enhancements. This helps maintainers guide your contributions effectively.
4. **Create a Branch**: Create a feature branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make Changes**: Implement your changes following the guidelines.
6. **Report reproduced evaluation metrics**: Ensure your changes work as expected by testing against publicly reported numbers. Add this information to [reproduced_benchmarks.md](reproduced_benchmarks.md).
7. **Commit Your Changes**: Write clear and concise commit messages:
   ```bash
   git commit -m "added new benchmark"
   ```
8. **Push Your Changes**: Push your branch to your fork:
   ```bash
   git push origin feature/your-benchmark-name
   ```
9. **Submit a Pull Request**: Open a pull request (PR) to the main repository.

## Issue Reporting

When reporting an issue, include:

1. A clear and descriptive title.
2. Steps to reproduce the issue.
3. Expected behavior vs. actual behavior.
4. Screenshots or logs, if applicable.
5. Environment details (e.g., OS, Python version).

## Pull Request Guidelines

1. Reference related issues in your PR description (e.g., "Closes #42").
2. Ensure your PR has descriptive and concise commits.
3. Keep your PR focused on a single change to make reviews easier.
4. Ensure your changes are rebased on the latest `main` branch:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
5. Address any feedback from maintainers promptly.

---

Thank you for contributing to Evalchemy! Your support and collaboration make this project better for everyone. 😊
