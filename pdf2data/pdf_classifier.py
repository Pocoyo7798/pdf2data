from typing import Any

from pydantic import BaseModel


class PDF_Classifier(BaseModel):
    document: Any

    def has_text(self) -> bool:
        """Identifies if a document has text in any page

        Returns
        -------
        bool
            True if the document has text in any page, False Otherwise
        """
        exist_text: bool = False
        # Run through all pages untils it detects a text block
        for page in self.document:
            # Get the area of text and images of the page
            exist_text = PDF_Classifier.page_has_text(page)
            if exist_text is True:
                break
        return exist_text

    @staticmethod
    def page_has_text(page: Any) -> bool:
        """Identifies if a document page has text
        Parameters
        ----------
        page : Any
            Document page to be evaluated

        Returns
        -------
        bool
            True if the page has text, False otherwise
        """
        has_text: bool = False
        # analyse all block in the page
        for block in page.get_text("blocks"):
            # verifies if the block is an image
            if "<image:" not in block[4]:
                has_text = True
                break
        return has_text
