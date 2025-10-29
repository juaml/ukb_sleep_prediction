#!/bin/bash

# To get the TSV file
# datalad clone ria+http://ukb.ds.inm7.de#~super
# cd super
# datalad get ukb668954.tsv

# 1160 -> Sleep duration
# 1170 -> Getting up in morning
# 1180 -> Morning/evening person (chronotype)
# 1190 -> Nap during day
# 1200 -> Sleeplessness / insomnia
# 1210 -> Snoring
# 1220 -> Daytime dozing / sleeping

# Basic Demographics
# 2178 -> Health rating
# 6138 -> Education
# 21000 -> Ethnic background
ukbb_parser parse --incsv /data/project/ukb_rls/data/ukb_phenotype/super/ukb668954.tsv -o /data/project/ukb_rls/data/ukb_phenotype/phenotypes --inhdr 1160 --inhdr 1170 --inhdr 1180 --inhdr 1190 --inhdr 1200 --inhdr 1210 --inhdr 1220 --inhdr 2178 --inhdr 6138 --inhdr 21000 --long_names --fillna NaN
