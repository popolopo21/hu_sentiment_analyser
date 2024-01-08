import logging
from typing import List, Optional
from tqdm import tqdm

# Constants for readability
RANGE_3 = 3
RANGE_5 = 5
RANGE_10 = 10

class TargetCleaner:
    """A class to clean and convert review scores between different ranges.
    
    Attributes:
        input_label_range (int): The range of the input labels.
        output_label_range (int): The desired range of the output labels.
        clean_rating (bool): Whether to clean the rating values.
        rating_at_end (bool): Whether the rating is at the end of the string.
        delimiter (Optional[str]): The delimiter used to split the rating from additional text.
    """
    
    def __init__(self, input_label_range: int, output_label_range: int, 
                 clean_rating: Optional[bool] = False, rating_at_end: Optional[bool] = False, delimiter: Optional[str] = None):
        if input_label_range not in {RANGE_3, RANGE_5, RANGE_10} or \
           output_label_range not in { RANGE_3, RANGE_5, RANGE_10}:
            raise ValueError("Invalid label range. Only 3, 5, and 10 are supported.")
        self.input_label_range = input_label_range
        self.output_label_range = output_label_range
        self.clean_rating = clean_rating
        self.rating_at_end = rating_at_end
        self.delimiter = delimiter

    
    def _convert_from_3_to_5(self, label: int) -> int:
        conversion_map = {1: 0, 2: 2, 3: 4}
        return conversion_map.get(label, label)
             
    def _convert_from_3_to_10(self, label: int) -> int:
        conversion_map = {1: 0, 2: 4, 3: 9}
        return conversion_map.get(label, label)
    
    def _convert_from_5_to_3(self, label: int) -> int:
        if label in {4, 5}:
            return 2
        elif label == 3:
            return 1
        else:
            return 0
    
    def _convert_from_5_to_10(self, label: int) -> int:
        conversion_map = {1: 0, 2: 2, 3: 4, 4: 7, 5: 9}
        return conversion_map.get(label, label)
    
    def _convert_from_10_to_3(self, label: int) -> int:
        if 7 <= label <= 10:
            return 2
        elif 4 <= label <= 6:
            return 1
        else:
            return 0
    
    def _convert_from_10_to_5(self, label: int) -> int:
        if label in {9, 10}:
            return 4
        elif label in {7, 8}:
            return 3
        elif label in {5, 6}:
            return 2
        elif label in {3, 4}:
            return 1
        else:
            return 0
    
    def _convert(self, label: int) -> int:
        """Converts a label from the input scale to the output scale."""
        if self.input_label_range == self.output_label_range:
            return label
        
        conversion_method = f'_convert_from_{self.input_label_range}_to_{self.output_label_range}'
        conversion_function = getattr(self, conversion_method, None)
        if conversion_function:
            return conversion_function(label)
        else:
            logging.warning(f"No conversion method found for {conversion_method}. Returning original label.")
            return label
    
    def _clean_rating(self, label: str) -> int:
        """Cleans the rating from a given string based on delimiter and position."""
        if label is None:
            return label
        if self.delimiter is None:
            try:
                label = int(label.strip())
                return label
            except ValueError:
                logging.error(f"Unable to convert {label} to an integer.")
                return None
            
        parts = label.split(self.delimiter)
        if len(parts) < 2:
            return label
        
        index = -1 if self.rating_at_end else 0
        try:
            cleaned_label = int(parts[index])
        except ValueError:
            logging.error(f"Unable to convert {parts[index]} to an integer.")
            return None
        return cleaned_label
    
    
    def process(self, labels: List[str]) -> List[int]:
        """Processes a list of labels, cleaning and converting them as necessary."""
        cleaned_labels = []
        for label in tqdm(labels, desc= "Processing labels"):
            try:
                if self.clean_rating:
                    label = self._clean_rating(label)
                if label is not None:
                    label = self._convert(int(label))
                if label is not None:
                    cleaned_labels.append(label)
            except Exception as e:
                cleaned_labels.append(None)
                raise Exception(f"Unable to convert {label} to an integer")
        return cleaned_labels