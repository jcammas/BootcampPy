from csvreader import CsvReader
if __name__ == "__main__":
    with CsvReader('good.csv') as file:
        if file == None:
            print("File is corrupted")
        else:
            data = file.getdata()
            header = file.getheader()
            print(header)
            print(data)
