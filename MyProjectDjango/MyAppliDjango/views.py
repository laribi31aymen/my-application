from django.shortcuts import render
from django.http import HttpResponse
from laribi_amghar import *
def ma_vue(request):

    resultat = laribi_amghar
    return HttpResponse(resultat)
# Create your views here.
