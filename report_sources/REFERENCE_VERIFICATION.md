# Reference verification

The bibliography uses the thirty supplied works plus the foundational FedAvg paper and the two dataset sources. DOI-bearing journal and book records were checked against publisher-deposited Crossref metadata. The NeurIPS and PMLR papers use their proceedings/preprint records; Lalitha et al. is cited as an arXiv preprint.

Corrections to the supplied list:

- Jiang et al.: DOI corrected to `10.1109/TWC.2022.3172147`.
- Chen et al., differential privacy: DOI corrected to `10.1109/TII.2022.3145010`.
- Li et al.: completed publication data, IEEE TCOM 72(11), 6973-6991 (2024), DOI `10.1109/TCOMM.2024.3407208`.
- Chung et al.: publication year corrected to **2026**, ACM Computing Surveys 58(8), DOI `10.1145/3785657`.
- Chen et al., mobility: confirmed DOI `10.1016/j.comnet.2025.111232` and article number 111232.
- Compression is treated separately from packet erasure; the supplied note calling compression a proxy for imperfect links is not adopted as a claim of equivalence.

`references.json` contains the verified bibliography. `reference_audit.json` maps the supplied entries to verified identifiers and years. `reference_numbering.json` records the report's first-citation numbering, which differs from the order of the supplied list.

The literature review is a targeted synthesis of the supplied collection, not an exhaustive systematic review. It does not claim that previous studies never considered these topics or that the project reimplements their algorithms.
