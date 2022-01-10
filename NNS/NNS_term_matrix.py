# -*- coding: utf-8 -*-
import pandas as pd


def NNS_term_matrix(x: pd.DataFrame, oos: [None, pd.Series] = None, names: bool = False) -> dict:
    r"""NNS Term Matrix

    Generates a term matrix for text classification use in \link{NNS.reg}.

    @param x mixed data.frame; character/numeric; A two column dataset should be used.  Concatenate text from original sources to comply with format.  Also note the possibility of factors in \code{"DV"}, so \code{"as.numeric(as.character(...))"} is used to avoid issues.
    @param oos mixed data.frame; character/numeric; Out-of-sample text dataset to be classified.
    @param names logical; Column names for \code{"IV"} and \code{"oos"}.  Defaults to FALSE.
    @return Returns the text as independent variables \code{"IV"} and the classification as the dependent variable \code{"DV"}.  Out-of-sample independent variables are returned with \code{"OOS"}.
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995/ref=cm_sw_su_dp}
    @examples
    x <- data.frame(cbind(c("sunny", "rainy"), c(1, -1)))
    NNS.term.matrix(x)

    ### Concatenate Text with space separator, cbind with "DV"
    x <- data.frame(cbind(c("sunny", "rainy"), c("windy", "cloudy"), c(1, -1)))
    x <- data.frame(cbind(paste(x[ , 1], x[ , 2], sep = " "), as.numeric(as.character(x[ , 3]))))
    NNS.term.matrix(x)


    ### NYT Example
    \dontrun{
    require(RTextTools)
    data(NYTimes)

    ### Concatenate Columns 3 and 4 containing text, with column 5 as DV
    NYT <- data.frame(cbind(paste(NYTimes[ , 3], NYTimes[ , 4], sep = " "),
                          as.numeric(as.character(NYTimes[ , 5]))))
    NNS.term.matrix(NYT)}
    @export
    """

    def count_unique(x1) -> list:
        return [x1.count(ii) for ii in unique_vocab]

    p = 0 if oos is None else len(oos)
    n = len(x)
    # Remove commas, etc.
    x.iloc[:, 0] = x.iloc[:, 0].astype(str)
    for i in [",", ";", ":", "'s", " . "]:
        x.iloc[:, 0] = x.iloc[:, 0].str.replace(i, "")

    unique_vocab = list(x.iloc[:, 0].str.split(" ", expand=True).stack().unique())

    # Sub with a longer .csv to be called to reduce IVs
    prepositions = [
        "a",
        "in",
        "of",
        "our",
        "the",
        "is",
        "for",
        "with",
        "we",
        "this",
        "it",
        "but",
        "was",
        "at",
        "to",
        "on",
        "aboard",
        "aside",
        "by",
        "means",
        "spite",
        "about",
        "as",
        "concerning",
        "instead",
        "above",
        "at",
        "considering",
        "into",
        "according",
        "atop",
        "despite",
        "view",
        "across",
        "because",
        "during",
        "near",
        "like",
        "across",
        "after",
        "against",
        "ahead",
        "along",
        "alongside",
        "amid",
        "among",
        "apart",
        "around",
        "out",
        "outside",
        "over",
        "owing",
        "past",
        "prior",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "besides",
        "between",
        "beyond",
        "regarding",
        "round",
        "since",
        "through",
        "throughout",
        "till",
        "down",
        "except",
        "from",
        "addition",
        "back",
        "front",
        "place",
        "regard",
        "inside",
        "together",
        "toward",
        "under",
        "underneath",
        "until",
        "nearby",
        "next",
        "off",
        "account",
        "onto",
        "top",
        "opposite",
        "out",
        "unto",
        "up",
        "within",
        "without",
        "what",
    ]

    # Remove prepositions
    preps = [i for i in unique_vocab if i in prepositions]

    if len(preps) > 0:
        unique_vocab = [i for i in unique_vocab if i not in prepositions]

    if oos is not None:
        oos_preps = [i for i in oos if i in prepositions]
        if len(oos_preps) > 0:
            oos = [i for i in oos if i not in prepositions]
        oos = pd.Series(oos)

    NNS_TM = x.iloc[:, 0].apply(count_unique).apply(pd.Series)
    # NNS_TM = t(
    #    sapply(
    #        1 : length(x[ , 1]),
    #        function(i) as.integer(
    #            tryCatch(
    #                stringr::str_count(x[i, 1], unique.vocab),
    #                error = function (e) 0
    #            )
    #        )
    #    )
    # )
    if names:
        NNS_TM.columns = unique_vocab

    if oos is not None:
        OOS_TM = oos.apply(count_unique).apply(pd.Series)
        # OOS_TM = t(
        #    sapply(
        #        1 : length(oos),
        #        function(i) stringr::str_count(oos[i], unique.vocab)
        #    )
        # )
        if names:
            OOS_TM.columns = unique_vocab

        return {"IV": NNS_TM, "DV": x.iloc[:, 1].values, "OOS": OOS_TM}

    return {"IV": NNS_TM, "DV": x.iloc[:, 1].values}


__all__ = ["NNS_term_matrix"]
