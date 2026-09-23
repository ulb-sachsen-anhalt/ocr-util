"""minidom_util module"""

from typing import Any, List
from xml.dom.minidom import Element, Node


class MinidomUtil:
    """Helper Methods for XML Manipulation"""

    @staticmethod
    def remove_element_and_clear_parent(element: Element, removable_tags=None) -> List[Element]:
        if removable_tags is None:
            removable_tags = []
        parent: Element = element.parentNode
        removed_elements: List[Element] = []
        if parent:
            parent.removeChild(element)
            removed_elements.append(element)
            siblings: List[Element] = parent.childNodes
            for sibling in siblings:
                is_text_node: bool = sibling.nodeType == Node.TEXT_NODE
                is_removable_tag: bool = sibling.nodeName in removable_tags
                is_removable: bool = is_text_node or is_removable_tag
                if is_removable:
                    parent.removeChild(sibling)
                    if is_removable_tag:
                        removed_elements.append(sibling)
            if len(parent.childNodes) == 0:
                removed_parent_elements = MinidomUtil.remove_element_and_clear_parent(parent, removable_tags)
                removed_elements.extend(removed_parent_elements)
        return removed_elements

    @staticmethod
    def set_attribute(element: Element, attr_name: str, value: Any) -> bool:
        attr_node: Node = element.getAttributeNode(attr_name)
        if attr_node is not None:
            old_value: str = str(attr_node.nodeValue)
            new_value: str = str(value)
            if new_value != old_value:
                attr_node.nodeValue = new_value
                return True
        return False
