# Contributing to MiniNN

>[!NOTE]
> This started as a personal project so probably there are some things that can be improved, feel free to open an issue or a pull request.

Thank you for your interest in contributing to **MiniNN**! Your support, whether through bug reports, feature suggestions, or documentation improvements, is greatly appreciated.

## How to contribute

There are several ways you can contribute to the project:

- **Reporting bugs**
- **Implementing features in the [TODO](TODO.md) list**
- **Suggesting new features**
- **Improving documentation**
- **Submitting pull requests**

## Development Setup

> [!IMPORTAT]
> All the changes should be made in the [`dev`](https://github.com/Pacatro/mininn/tree/dev) branch, so please create a new branch for your changes and submit a pull request to that branch.

To set up the project locally for development, follow these steps:

1. Clone the `dev` branch of the repository:

    ```bash
    git clone -b dev https://github.com/Pacatro/mininn.git
    ```

2. Install dependencies and set up the environment, there is a `makefile` that can be used for this:

    ```bash
    cd mininn
    make
    ```

3. Run the tests to make sure everything is working:

    ```bash
    cargo test
    ```

## Style Guidelines

- **Code formatting**: Please use the `rustfmt` tool to format your code.

    ```bash
    cargo fmt
    ```

- **Documentation**: Ensure that your code is well-documented, including doc comments for public functions, structs, and methods. You can check the documentation for the project by running:

    ```bash
    cargo doc --open
    ```

- **Testing**: Write tests for new features and bug fixes. Run tests locally before submitting a pull request.
