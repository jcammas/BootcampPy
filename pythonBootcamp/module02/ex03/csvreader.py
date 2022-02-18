class CsvReader():
    def __init__(self, name='', sep=',', header=False, skip_top=0, skip_bottom=0):
        self.name = name
        self.sep = sep
        self.header = header
        self.skip_top = skip_top
        self.skip_bottom = skip_bottom
        self.file = []
        self.raw_file = ''
        self.raw_lines = []
        self.corrupted = 0
        self.line = []

    def __enter__(self):
        self.fd = open(self.name, 'r')
        self.raw_file = self.fd.read()
        self.raw_file = self.raw_file[:-1]
        if (self.raw_file):
            self.raw_lines = self.raw_file.split('\n')
            for line in self.raw_lines:
                self.file.append(line.split(self.sep))
                self.line.append([])
            for i in range(len(self.file)):
                for elem in self.file[i]:
                    self.line[i].append(elem.strip())
            length = len(self.line[1])
            for line in self.line:
                if (len(line) != length):
                    self.corrupted = 1
                for elem in line:
                    if elem == None or elem == '':
                        self.corrupted = 1
        else:
            self.raw_lines = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fd.close()

    def getheader(self):
        if (self.corrupted):
            return None
        else:
            return self.line[0]

    def getdata(self):
        if (self.corrupted):
            return None
        else:
            return self.line[1:]

# https://docs.python.org/fr/3.6/library/csv.html
