"""STORM extension modules."""
from storm_modules.extraction_store import save_citations, save_figure_results, save_table_results
from storm_modules.figure_extractor import DeepFiguresExtractor as FigureExtractor
from storm_modules.openalex_client import OpenAlexClient
from storm_modules.pars_cit import ParsCitExtractor
from storm_modules.table_extractor import TableExtractor
from storm_modules.theory_builder import TheoryDatabaseBuilder
from storm_modules.hypothesis_generator import HypothesisGenerator
from storm_modules.study_designer import QuantitativeStudyDesigner
from storm_modules.literature_synthesizer import LiteratureSynthesizer
from storm_modules.academic_writer import AcademicWriter
from storm_modules.thesis_generator import ThesisGenerator
from storm_modules.knowledge_graph import KnowledgeGraphManager
from storm_modules.icite_client import ICiteClient
from storm_modules.s2orc_client import S2ORCClient
from storm_modules.keyphrase_extractor import KeyphraseExtractor

__all__ = [
    "save_citations",
    "save_figure_results",
    "save_table_results",
    "FigureExtractor",
    "OpenAlexClient",
    "ParsCitExtractor",
    "TableExtractor",
    "TheoryDatabaseBuilder",
    "HypothesisGenerator",
    "QuantitativeStudyDesigner",
    "LiteratureSynthesizer",
    "AcademicWriter",
    "ThesisGenerator",
    "KnowledgeGraphManager",
    "ICiteClient",
    "S2ORCClient",
    "KeyphraseExtractor"
]
