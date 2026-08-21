classdef PropMultiChannelIndependence < matlab.unittest.TestCase
%PROPMULTICHANNELINEPENDENCE  Matrix input is processed column-independently.
%
%   When filterbank receives an L × W matrix, it applies the filterbank
%   independently to each column.  This test verifies:
%
%   (1) filterbank([x₁ x₂], g, a){m}[:, w] == filterbank(xw, g, a){m}
%   (2) Output coefficient arrays have the correct W-column shape.
%   (3) Linear combinations across channels are handled correctly.

    properties
        p   % scalar parameters (fs, Ls)
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);
            tc.p = struct('fs', 8000, 'Ls', 1024);
        end
    end

    methods (Test)

        function testTwoColumnInputMatchesSeparateCalls(tc)
            % filterbank([x1,x2], g, a){m} must equal [filterbank(x1,g,a){m}, filterbank(x2,g,a){m}].
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);

            for trial = 1:50
                x1 = randn(L, 1);
                x2 = randn(L, 1);

                c12 = filterbank([x1, x2], g, a);
                c1  = filterbank(x1, g, a);
                c2  = filterbank(x2, g, a);

                M = numel(g);
                maxErr = 0;
                for m = 1:M
                    e1 = norm(c12{m}(:,1) - c1{m}) / (norm(c1{m}) + eps);
                    e2 = norm(c12{m}(:,2) - c2{m}) / (norm(c2{m}) + eps);
                    maxErr = max(maxErr, max(e1, e2));
                end
                tc.verifyLessThan(maxErr, 1e-12, ...
                    sprintf('Trial %d: multi-channel independence error %.2e', trial, maxErr));
            end
        end

        function testOutputHasCorrectChannelDimension(tc)
            % c{m} must have W columns when input is L × W.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);

            for W = [1, 2, 3, 5]
                x = randn(L, W);
                c = filterbank(x, g, a);
                M = numel(g);
                for m = 1:M
                    tc.verifyEqual(size(c{m}, 2), W, ...
                        sprintf('W=%d, band %d: expected %d channels, got %d', ...
                        W, m, W, size(c{m}, 2)));
                end
            end
        end

        function testLinearCombinationAcrossChannels(tc)
            % filterbank(α x₁ + β x₂, g, a) == α filterbank(x₁,g,a) + β filterbank(x₂,g,a)
            % This tests cross-channel linearity from the single-column perspective.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);

            alpha = 2.5;
            beta  = -1.3i;

            for trial = 1:30
                x1 = randn(L, 1);
                x2 = randn(L, 1) + 1i*randn(L, 1);

                c1  = filterbank(x1, g, a);
                c2  = filterbank(x2, g, a);
                ccc = filterbank(alpha*x1 + beta*x2, g, a);

                M      = numel(g);
                maxErr = 0;
                for m = 1:M
                    denom = norm(c1{m}) + norm(c2{m}) + eps;
                    err   = norm(ccc{m} - (alpha*c1{m} + beta*c2{m})) / denom;
                    maxErr = max(maxErr, err);
                end
                tc.verifyLessThan(maxErr, 1e-10, ...
                    sprintf('Trial %d: cross-channel linearity error %.2e', trial, maxErr));
            end
        end

        function testThreeColumnInputMatchesSeparateCalls(tc)
            % Extend independence check to W = 3 columns.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);

            x1 = randn(L, 1);
            x2 = randn(L, 1) + 1i*randn(L, 1);
            x3 = randn(L, 1);

            c123 = filterbank([x1, x2, x3], g, a);
            c1   = filterbank(x1, g, a);
            c2   = filterbank(x2, g, a);
            c3   = filterbank(x3, g, a);

            M = numel(g);
            for m = 1:M
                e1 = norm(c123{m}(:,1) - c1{m}) / (norm(c1{m}) + eps);
                e2 = norm(c123{m}(:,2) - c2{m}) / (norm(c2{m}) + eps);
                e3 = norm(c123{m}(:,3) - c3{m}) / (norm(c3{m}) + eps);
                tc.verifyLessThan(max([e1,e2,e3]), 1e-12, ...
                    sprintf('Band %d: 3-column independence error (max %.2e)', m, max([e1,e2,e3])));
            end
        end

    end
end
