#!/usr/bin/env perl
#-*-perl-*-
#
# convert attention matrix to cvs files
# to be used with plot-attention.py
#
# OPTIONS
#   -s <sentnr> ..... attention of sentence number <sentnr>
#   -p .............. keep padding (this is ignored otherwise)


use strict;
use Getopt::Std;

our ($opt_s,$opt_p);
getopts("ps:");

my $sent = $opt_s || 0;
my $keepPadding = $opt_p;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

my @words = ();
my $sentCount = 0;
my $wordCount = 0;

print join(',',('source','target','attention'));
print "\n";

while (<>){
    chomp;
    unless (@words){
	@words = split(/\t/);
	map($_=~s/,/COMMA/,@words);
	shift(@words);
	next;
    }
    if (/\S/){
	$wordCount++;
	if ($sentCount == $sent){
	    my @attn = split(/\t/);
	    my $src = shift(@attn);
	    $src=~s/,/COMMA/;

	    unless ($keepPadding){
		next if ($src eq '_');
		pop(@attn);
	    }
	    
	    foreach (0..$#attn){
		printf("%03d_%s,%03d_%s,%f\n",$wordCount,$src,$_,$words[$_],$attn[$_]);
	    }
	}
    }
    else{
	$sentCount++;
	$wordCount=0;
	@words=();
    }
    last if ($sentCount>$sent);
}
