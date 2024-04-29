from abc import ABC, abstractmethod

class SensorService(ABC): 
    @abstractmethod 
    def terminate(self) -> None: 
        pass 
 
    def is_valid(self) -> bool: 
        return True 
    
    def execute(self) -> None: 
        pass 