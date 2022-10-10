## Response to previous comments

The last submission was returned because the file R/pdsoftbart.R has commands
that printed to the console. I think I fixed it (by using warning() rather than
cat()). 

## R CMD check results

There were no ERRORs or WARNINGs. There was one NOTE:

New submission

Possibly misspelled words in DESCRIPTION:
  BayesTree (9:227)
  Linero (9:60)
  SoftBart (3:39, 9:29)
  
These words are not misspelled.

## Downstream dependencies

There are currently no downstream dependencies for this package.