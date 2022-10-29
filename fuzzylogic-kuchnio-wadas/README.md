# Date score possibility [Fuzzy Logic]
- zjazd 2 [22.10.2022] (NSI, semestr 7)
 * Authors :
   - Karol Kuchnio (`s21912@pjwstk.edu.pl`)
   - Michał Wadas (`s20495@pjwstk.edu.pl`)


# INSTALLATION
    - pip install matplotlib
    - pip install scikit-fuzzy


#  RULES
Antecednets (Inputs): 
* height
* appearance
* iq

Consequents (Outputs): 

* successful date possibility

1.  if height is `high` OR appearance is `high` OR iq is `high`, then successful date possibility is `high`
2.  if height is `average` OR appearance is `average` OR iq is `average`, then successful date possibility is `average`
3.  if height is `low` OR appearance is `low` OR iq is `low`, then successful date possibility is `low`
