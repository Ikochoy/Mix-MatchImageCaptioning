import re
class AnnotationCleaner:
    def __init__(self, annotations):
        self.annotations = annotations
        self.clean_annotations = []
        for annotation in self.annotations:
            self.clean_annotations.append(self.__call__(annotation))
        return self.clean_annotations

    def removePunctuation(self, annotation):
        # remove the punctuations
        return re.sub(r'[^\w\s]', '', annotation)

    def lower(self, annotation):
        # convert annotations to lower case
        return annotation.lower()
    
    def removeExtraSpace(self, annotation):
        # remove extra spaces
        return re.sub(r" {2,}", " ", annotation)

    def __call__(self, annotation):
        output = self.removePunctuation(annotation)
        output = self.lower(annotation)
        output = self.removeExtraSpace(annotation)
        return output

