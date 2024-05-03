def text_to_bin(infile_name: str, outfile_name: str) -> None:
    with open(infile_name, 'r') as infile:
        with open(outfile_name, 'wb') as outfile:
            for line in infile:
                for num in line.split():
                    outfile.write(int(num).to_bytes(length=4, byteorder='little'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='gconverter',
                                     description='Convert between graph formats')
    parser.add_argument('-i', '--infile', required=True)
    parser.add_argument('-o', '--outfile', required=True)

    args = parser.parse_args()

    text_to_bin(args.infile, args.outfile)
