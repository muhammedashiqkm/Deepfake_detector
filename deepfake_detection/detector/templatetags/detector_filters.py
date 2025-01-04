from django import template

register = template.Library()

@register.filter(name='multiply')
def multiply(value, arg):
    """Multiply the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter(name='index')
def index(lst, i):
    """Get an item from a list by index"""
    try:
        return lst[i]
    except (IndexError, TypeError):
        return ""
