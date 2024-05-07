from django.shortcuts import render
from django.http import HttpResponse
import laribi_amghar
def mon_endpoint(request):
    resultat = laribi_amghar
    return HttpResponse(resultat)

# Create your views here.
