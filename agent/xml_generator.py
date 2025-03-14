# agent/xml_generator.py
import lxml.etree as ET

class XMLGenerator:
    def __init__(self, xsd_path: str = "agent/schemas/scenario.xsd"):
        self.xsd_path = xsd_path
        # Load and parse the XSD for validation
        self.schema = None
        self._load_xsd()

    def _load_xsd(self):
        try:
            with open(self.xsd_path, 'rb') as f:
                xmlschema_doc = ET.parse(f)
                self.schema = ET.XMLSchema(xmlschema_doc)
        except Exception as e:
            print(f"Warning: Unable to load or parse XSD. Validation might be disabled. Error: {e}")
            self.schema = None

    def build_xml(self, scenario_data: str, user_requirements: str, additional_metadata: str = "") -> str:
        """
        Builds a multi-layer XML with potential attributes and multiple elements.
        scenario_data might come from RAG, user_requirements is the user request,
        and additional_metadata can be any extra info (like a summary or historical data).
        """
        root = ET.Element("ScenarioOutput", attrib={"version": "1.0"})

        scenario_elem = ET.SubElement(root, "ScenarioContext")
        scenario_elem.text = scenario_data

        user_req_elem = ET.SubElement(root, "UserRequirements")
        user_req_elem.text = user_requirements

        if additional_metadata:
            meta_elem = ET.SubElement(root, "AdditionalMetadata")
            meta_elem.text = additional_metadata

        # Convert to string
        xml_string = ET.tostring(root, pretty_print=True, encoding="UTF-8").decode("UTF-8")

        # Validate against XSD if available
        if self.schema:
            is_valid, error_str = self.validate_xml(xml_string)
            if not is_valid:
                raise ValueError(f"Generated XML is invalid against the XSD schema. Error: {error_str}")

        return xml_string

    def validate_xml(self, xml_string: str):
        """
        Validates the given XML string against the loaded XSD.
        Returns (bool, error_msg).
        """
        try:
            xml_doc = ET.fromstring(xml_string.encode("utf-8"))
            self.schema.assertValid(xml_doc)
            return True, ""
        except ET.DocumentInvalid as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
