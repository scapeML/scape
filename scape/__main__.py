def main():
    # Show the name and version of the package
    import scape

    print(f"Running ScAPE version {scape.__version__}")
    # Show the help message
    import argparse

    parser = argparse.ArgumentParser(
        description="ScAPE - Single Cell Analysis of Perturbational Effects"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"scape version {scape.__version__}",
        help="Show the version of scape",
    )
    parser.parse_args()


if __name__ == "__main__":
    main()