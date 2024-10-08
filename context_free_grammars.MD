Basic Single Motif Grammar: Rule: S → N*[TF]N* Example: AATTC[GATA1]GGTCA

Fixed-Space Two-Motif Grammar: Rule: S → N*[TF1]N^k[TF2]N* Example: AATT[AP1]NNNNNN[GR]GGTCA

Variable-Space Two-Motif Grammar: Rule: S → N*[TF1]N*[TF2]N* Example: AATT[AP1]NNNNNN[GR]GGTCA or AATT[AP1][GR]GGTCA

Or-Logic Grammar: Rule: S → N*([TF1]|[TF2])N* Example: AATTC[GATA1]GGTCA or AATTC[TAL1]GGTCA

Repetitive Motif Grammar: Rule: S → N*([TF]N*)* Example: AATT[CTCF]NNN[CTCF]NNNNN[CTCF]GGTCA

Palindromic Motif Grammar: Rule: S → N*[TF]N*[TF_reverse]N* Example: AATT[GATA→]NNNN[←ATAG]GGTCA

Ordered Multi-Motif Grammar: Rule: S → N*[TF1]N*[TF2]N*[TF3]N* Example: AATT[GATA1]NNNN[TAL1]NNNNNNNN[RUNX1]GGTCA

Alternating Motif Grammar: Rule: S → N*([TF1]N*[TF2]N*)* Example: AATT[DORSAL]NNN[SNAIL]NNNNN[DORSAL]NNN[SNAIL]GGTCA

Nested Motif Grammar: Rule: S → N*[TF1]NS[TF1]N Example: AATT[CTCF]NNN[GATA1]NNN[GATA1]NNN[CTCF]GGTCA

Complex Multi-Motif Grammar: Rule: S → N*([TF1]N*[TF2]N*|[TF3]N*[TF4]N*)([TF5]|[TF6])N Example: AATT[PU1]NN[IRF8]NNN[GATA1]NNNN[TAL1]NN[OCT4]GGTCA

In enhancer biology, various motif grammars help describe the arrangement and interaction of transcription factor (TF) binding sites. The simplest form is the Single Motif Grammar, which features a single TF binding site. Expanding on this, the Fixed-Space Two-Motif Grammar represents cooperative interactions between two TFs with a fixed distance between them, while the Variable-Space Two-Motif Grammar allows for flexible spacing, which is common in enhancers. The Or-Logic Grammar models enhancers that can be activated by either of two different TFs. For repetitive structures, the Repetitive Motif Grammar captures homotypic clusters of TF binding sites, frequently seen in enhancer regions. Some TF binding sites occur in palindromic orientations, especially with nuclear receptors, represented by the Palindromic Motif Grammar. More complex arrangements include the Ordered Multi-Motif Grammar, where multiple TFs bind in a specific sequence, and the Alternating Motif Grammar, which involves a repeating pattern of different TF binding sites. The Nested Motif Grammar is used for more intricate enhancers where one TF’s binding site flanks another, and the Complex Multi-Motif Grammar represents highly complex enhancers with multiple TFs and variable spacing. The TF motifs chosen for these grammars span a wide range of biological processes: GATA1, TAL1, RUNX1 are involved in blood cell development, AP1 and GR in stress response, CTCF in chromatin organization, and PU1, IRF8 in immune development. Other important TFs include OCT4 for stem cell pluripotency, ETS for cell proliferation, SREBP for lipid metabolism, CREB for cAMP responses, STAT for cytokine signaling, and DREF, TRL, E2F for chromatin remodeling and cell cycle regulation, especially in Drosophila.
