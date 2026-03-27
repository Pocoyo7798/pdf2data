from typing import Any, Dict, List, Optional

import fitz
import importlib_resources

from pdf2data.table_extractor import TableExtractor

TABLE1 = {
        "number": 1,
        "caption": "Multi-row header test",
        "column_headers": [0, 1],
        "block": [
            ["",       "Surface area", "Pore volume",    "Pore volume"   ],
            ["Sample", "SBET (m2/g)",  "Vmeso (cm3/g)",  "Vmicro (cm3/g)"],
            ["Cat-A",  "450",          "0.10",            "0.05"          ],
            ["Cat-B",  "380",          "0.08",            "0.04"          ],
        ],
}
TABLE1_RESULTS = [
    {'sample': 'Cat-A ', 'external_area': '450 m2/g', 'mesopore_volume': '0.10 cm3/g', 'micropore_volume': '0.05 cm3/g'}, 
    {'sample': 'Cat-B ', 'external_area': '380 m2/g', 'mesopore_volume': '0.08 cm3/g', 'micropore_volume': '0.04 cm3/g'}
    ]

TABLE2 = {
        "number": 2,
        "caption": "Inverted table test",
        "column_headers": [0],
        "block": [
            ["Property",         "Cat-A", "Cat-B"],
            ["Sample",           "Cat-A", "Cat-B"],
            ["SBET (m2/g)",      "450",   "380"  ],
            ["Vmeso (cm3/g)",    "0.10",  "0.08" ],
            ["Vmicro (cm3/g)",   "0.05",  "0.04" ],
        ],
    }
TABLE2_RESULTS = [
    {'sample': 'Cat-A ', 'surface_area': '450 m2/g', 'mesopore_volume': '0.10 cm3/g', 'micropore_volume': '0.05 cm3/g'}, 
    {'sample': 'Cat-B ', 'surface_area': '380 m2/g', 'mesopore_volume': '0.08 cm3/g', 'micropore_volume': '0.04 cm3/g'}
    ]
def test_table_extractor():
    extractor = TableExtractor(table_type="characterization")
    table1_result = extractor.extract_table(TABLE1)
    table1_list = [
                            {key: cv.value + " " + cv.unit
                             for key, cv in row.data.items()}
                            for row in table1_result.rows
                        ]
    table2_result = extractor.extract_table(TABLE2)
    table2_list = [
                            {key: cv.value + " " + cv.unit
                             for key, cv in row.data.items()}
                            for row in table2_result.rows
                        ]
    assert table1_list == TABLE1_RESULTS
    assert table2_list == TABLE2_RESULTS
    
