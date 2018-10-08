from abc import ABC, abstractmethod

class TemplateBaseClass(ABC):
    """ 
    This is an abstract class for all the templates. 
    """
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def build_table(self):
        pass
    
    @abstractmethod
    def load_table(self):
        pass
    
    @abstractmethod
    def get_score(self):
        pass
       
    

    