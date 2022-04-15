import re
class AnnotationCleaner:
    def __init__(self, annotations):
        self.annotations = annotations

    def clean(self):
        self.clean_annotations = []
        for annotation in self.annotations:
            self.clean_annotations.append(self.run(annotation))
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

    def run(self, annotation):
        output = self.removePunctuation(annotation)
        output = self.lower(annotation)
        output = self.removeExtraSpace(annotation)
        return output

    def get_cleaned(self):
      cleaned = []
      for annotation in self.annotations:
        cleaned.append(self.run(annotation))
      return cleaned

