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
    def dump_data(self,filename):
        pass
    
    @abstractmethod
    def load_table(self,filename):
        pass
    
    @abstractmethod
    def compute_score(self,triple):
        pass
    @abstractmethod
    def get_input(self,fact):
       
    

    