# Portfolio restoration notes

The original creative-piece ZIP identifies this as 6G6Z0019 Synoptic Project and contains files dated July 2025. The historical report's MSc cover wording conflicts with that identification; use the unit name and year above.

The public notebook had been reduced to three lines. It has been restored from the submitted 58-cell notebook. Its explanations have been shortened and corrected; saved numerical outputs are retained as historical evidence, not represented as newly executed results. Student-number text was removed without deleting whole lines.

The report and original narrative claimed temperature scaling. The supplied code computes ECE from averaged probabilities but contains no temperature fitting step. Its final JSON also stored a missing ECE despite a notebook output of 0.015280134256005383. Accordingly, the maintained description says confidence evaluation, not calibrated probabilities. The historical PDF is retained unchanged and its stronger clinical, calibration and generalisation statements should not be treated as verified findings.

The source notebook contains an optional check for `probs_cal`; that does not establish that calibrated predictions were generated. The final saved variant is `ensemble`.

The split is at image level. Different paths do not prove different patients or different image content. The earlier validation-as-test fallback has been removed from the maintained loader. Original saved code remains visible for provenance.

Restoration checks cover notebook structure, Python syntax and split/metric consistency. Training and checkpoint-based reproduction have not been performed.
