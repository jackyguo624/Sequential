from pytorch_lightning.cli import LightningCLI, ArgsType

def cli_main(args: ArgsType = None):
    cli = LightningCLI(args=args)


if __name__ == "__main__":
    cli_main()