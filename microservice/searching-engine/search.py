import csv

class SearchingEngine():
    def __init__(self, index_csv_file):
        """
        """
        self.index_csv_file = index_csv_file
        self.index = self.create_index(index_csv_file)

    def create_index(self, index_csv_file):
        index_dict = {}
        with open(index_csv_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    print(f'\tSimilar items to {row[0]} are {row[1]}.')
                    index_dict[row[0]] = eval(row[1])
                    line_count += 1
            print(f'Processed {line_count} lines.')
        return index_dict
