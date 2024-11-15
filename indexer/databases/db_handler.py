from abc import abstractmethod


class DBHandler:
    @abstractmethod
    def connect(self, **kwargs):
        pass

    @abstractmethod
    def create_db(self, **kwargs):
        pass

    @abstractmethod
    def add_records(self, **kwargs):
        pass

    @abstractmethod
    def delete_item(self, **kwargs):
        pass

    @abstractmethod
    def bulk_index(self, **kwargs):
        pass