classdef PropFilterbanklengthProperties < matlab.unittest.TestCase
%PROPFILTERBANKLENGTPROPERTIES  Numeric properties of filterbanklength.
%
%   filterbanklength(Ls, a) returns the smallest integer L >= Ls such that
%   L is divisible by lcm(a(:,1)).  The following properties are tested:
%
%   (1) Output >= Ls          (length covers the signal)
%   (2) Idempotence           (applying filterbanklength twice gives the same L)
%   (3) Divisibility by a     (for uniform a: mod(L, a) == 0)
%   (4) Divisibility by lcm   (for non-uniform a: mod(L, lcm(a)) == 0)
%   (5) Monotone in Ls        (non-decreasing as Ls increases)
%   (6) L == Ls when Ls is already divisible by lcm(a)

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    methods (Test)

        function testLengthAtLeastLs(tc)
            for Ls = [1, 100, 511, 512, 1000, 1023, 1024, 2048, 4095]
                for a = [1, 2, 4, 8, 16, 32]
                    L = filterbanklength(Ls, a);
                    tc.verifyGreaterThanOrEqual(L, Ls, ...
                        sprintf('filterbanklength(%d,%d)=%d < Ls', Ls, a, L));
                end
            end
        end

        function testIdempotence(tc)
            % filterbanklength(filterbanklength(Ls,a), a) == filterbanklength(Ls,a)
            for Ls = [100, 512, 1000, 1023, 4095]
                for a = [1, 2, 4, 8, 16]
                    L1 = filterbanklength(Ls, a);
                    L2 = filterbanklength(L1, a);
                    tc.verifyEqual(L1, L2, ...
                        sprintf('Not idempotent: filterbanklength(%d,%d)=%d, second call=%d', ...
                        Ls, a, L1, L2));
                end
            end
        end

        function testDivisibilityUniformHop(tc)
            % mod(filterbanklength(Ls, a), a) == 0 for scalar a.
            for Ls = [100, 512, 1000, 1023, 4095]
                for a = [1, 2, 4, 8, 16, 32]
                    L = filterbanklength(Ls, a);
                    tc.verifyEqual(mod(L, a), 0, ...
                        sprintf('filterbanklength(%d,%d)=%d not divisible by a=%d', ...
                        Ls, a, L, a));
                end
            end
        end

        function testDivisibilityByLcmNonuniformHop(tc)
            % For a non-uniform hop vector, L must be divisible by lcm(a).
            Ls     = 1024;
            a_sets = {[2;4], [3;6], [4;8;16], [2;3;4], [6;10;15]};
            for k = 1:numel(a_sets)
                a     = a_sets{k};
                L     = filterbanklength(Ls, a);
                a_lcm = a(1);
                for j = 2:numel(a)
                    a_lcm = lcm(a_lcm, a(j));
                end
                tc.verifyEqual(mod(L, a_lcm), 0, ...
                    sprintf('a=%s: L=%d not divisible by lcm=%d', ...
                    mat2str(a'), L, a_lcm));
            end
        end

        function testMonotoneInLs(tc)
            % filterbanklength is non-decreasing as Ls increases.
            for a = [4, 8, 16]
                prev = 0;
                for Ls = 1 : 50 : 1000
                    L = filterbanklength(Ls, a);
                    tc.verifyGreaterThanOrEqual(L, prev, ...
                        sprintf('a=%d: not monotone at Ls=%d (prev=%d, now=%d)', ...
                        a, Ls, prev, L));
                    prev = L;
                end
            end
        end

        function testExactLengthWhenAlreadyDivisible(tc)
            % If Ls is already divisible by a, filterbanklength(Ls, a) == Ls.
            for a = [2, 4, 8, 16]
                for k = 1:10
                    Ls = a * k * 16;
                    L  = filterbanklength(Ls, a);
                    tc.verifyEqual(L, Ls, ...
                        sprintf('a=%d, Ls=%d already divisible, but L=%d != Ls', a, Ls, L));
                end
            end
        end

    end
end
